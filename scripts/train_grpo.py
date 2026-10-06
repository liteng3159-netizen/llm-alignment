import json
import sys

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import wandb

from llm_alignment.math_utils import extract_answer
from llm_alignment.grpo import grpo_train_step


device = torch.device("cuda:0")




# ============================================================
# 2. Model
# ============================================================

model_name = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Qwen/Qwen2.5-3B-Instruct"
)

print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
    trust_remote_code=True,
    attn_implementation="sdpa",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model.train()


# ============================================================
# 3. WandB
# ============================================================

wandb.init(
    project=f"{model_name.replace('/', '-')}-my-grpo-gsm8k"
)


# ============================================================
# 4. Load dataset
# ============================================================

PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Assistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer>
User: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Assistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer>
User: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Assistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer>
User: {question}
Assistant: <think>"""

train_data = []

with open("data/gsm8k/train.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)

        question = item["question"]
        answer = item["answer"]

        # 从 GSM8K 原始 answer 中提取最终数值答案
        numerical_answer = extract_answer(answer)

        # 构造完整 prompt
        prompt = PROMPT_TEMPLATE.format(
            question=question
        )

        train_data.append(
            {
                "prompt": prompt,
                "numerical_answer": numerical_answer,
            }
        )

print(f"Loaded {len(train_data)} samples")


# ============================================================
# 5. Filter long prompts
# ============================================================

MAX_PROMPT_CHARS = 2000

train_data = [
    item
    for item in train_data
    if len(item["prompt"]) <= MAX_PROMPT_CHARS
]

print(
    f"After length filter: {len(train_data)} samples"
)

dataset = Dataset.from_list(train_data)


# ============================================================
# 6. Reward function
# ============================================================

def reward_fn(
    response: str,
    ground_truth: int,
) -> dict[str, float]:

    # --------------------------------------------------------
    # accuracy reward
    # --------------------------------------------------------

    predicted = extract_answer(response)

    if predicted == ground_truth:
        accuracy_reward = 1.0
    else:
        accuracy_reward = -1.0

    # --------------------------------------------------------
    # format reward
    # --------------------------------------------------------

    has_think = (
        "</think>" in response
    )

    has_answer = (
        "<answer>" in response
        and "</answer>" in response
    )

    if has_think and has_answer:
        format_reward = 0.2
    else:
        format_reward = -0.2

    # --------------------------------------------------------
    # length reward
    # --------------------------------------------------------

    if len(response) < 30:
        length_reward = -0.5
    else:
        length_reward = 0.0

    # --------------------------------------------------------
    # total reward
    # --------------------------------------------------------

    total_reward = (
        accuracy_reward
        + format_reward
        + length_reward
    )

    return {
        "reward": total_reward,
        "format_reward": format_reward,
    }

# ============================================================
# 7. Optimizer
# ============================================================

learning_rate = 5e-6

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
)


# ============================================================
# 8. GRPO hyperparameters
# ============================================================

num_train_epochs = 1

group_size = 4

# 每次 rollout 有多少个 prompt
num_prompts_per_rollout = 2

# 每个 prompt 生成 4 个 response
num_generations = group_size

# 总 response 数量
rollout_batch_size = (
    num_prompts_per_rollout
    * num_generations
)

# 你的 grpo_train_step 内部会把
#
# rollout_batch_size
# /
# gradient_accumulation_steps
#
# 当作 microbatch size
#
gradient_accumulation_steps = 4

max_grad_norm = 1.0

max_new_tokens = 512

temperature = 0.7

top_p = 1.0


# ============================================================
# 9. Sanity check
# ============================================================

assert (
    rollout_batch_size % group_size == 0
)

assert (
    rollout_batch_size
    % gradient_accumulation_steps
    == 0
)

print()
print("========== GRPO CONFIG ==========")
print(f"num prompts / rollout: {num_prompts_per_rollout}")
print(f"num generations:       {num_generations}")
print(f"group size:            {group_size}")
print(f"rollout batch size:    {rollout_batch_size}")
print(
    f"gradient accumulation: "
    f"{gradient_accumulation_steps}"
)
print(
    f"microbatch size:       "
    f"{rollout_batch_size // gradient_accumulation_steps}"
)
print(f"learning rate:         {learning_rate}")
print(f"max new tokens:        {max_new_tokens}")
print("=================================")
print()


# ============================================================
# 10. Training
# ============================================================

global_step = 0

for epoch in range(num_train_epochs):

    print(f"\n========== Epoch {epoch + 1} ==========\n")

    # --------------------------------------------------------
    # shuffle
    # --------------------------------------------------------

    indices = torch.randperm(
        len(dataset)
    ).tolist()

    epoch_loss = 0.0
    num_steps = 0

    # --------------------------------------------------------
    # iterate prompts
    # --------------------------------------------------------

    for start_idx in range(
        0,
        len(indices),
        num_prompts_per_rollout,
    ):

        prompt_indices = indices[
            start_idx:
            start_idx + num_prompts_per_rollout
        ]

        # 最后不足一个 batch 就丢掉
        if len(prompt_indices) < num_prompts_per_rollout:
            break

        prompts = [
            dataset[i]["prompt"]
            for i in prompt_indices
        ]

        ground_truths = [
            dataset[i]["numerical_answer"]
            for i in prompt_indices
        ]

        # ====================================================
        # 11. Generate
        # ====================================================

        # tokenize prompts
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        prompt_input_ids = (
            prompt_inputs["input_ids"]
            .to(device)
        )

        prompt_attention_mask = (
            prompt_inputs["attention_mask"]
            .to(device)
        )

        # ----------------------------------------------------
        # 每个 prompt 生成 4 个 response
        # ----------------------------------------------------

        with torch.no_grad():

            generated = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,

                max_new_tokens=max_new_tokens,

                do_sample=True,
                temperature=temperature,
                top_p=top_p,

                num_return_sequences=num_generations,

                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # ====================================================
        # 12. Extract responses
        # ====================================================

        rollout_responses = []

        repeated_prompts = []

        repeated_ground_truths = []

        # generate() 的输出前 input_width 个 token
        # 是 padding 后的 prompt
        input_width = prompt_input_ids.shape[1]

        for prompt_idx in range(len(prompts)):

            prompt = prompts[prompt_idx]
            ground_truth = ground_truths[prompt_idx]

            for generation_idx in range(num_generations):

                global_generation_idx = (
                    prompt_idx * num_generations
                    + generation_idx
                )

                generated_ids = generated[
                    global_generation_idx
                ]

                # --------------------------------------------
                # 去掉整个 padded prompt
                # --------------------------------------------

                response_ids = generated_ids[
                    input_width:
                ]

                response = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                )

                rollout_responses.append(response)

                repeated_prompts.append(prompt)

                repeated_ground_truths.append(
                    ground_truth
                )

        # ====================================================
        # 13. Sanity check
        # ====================================================

        assert len(
            rollout_responses
        ) == rollout_batch_size

        assert len(
            repeated_prompts
        ) == rollout_batch_size

        assert len(
            repeated_ground_truths
        ) == rollout_batch_size

        # ====================================================
        # 14. Print samples
        # ====================================================


        print("\n------------------------------")

        print(
            f"Prompt:\n"
            f"{prompts[0]}"
        )

        for i in range(
            num_generations
        ):

            print(
                f"\nGeneration {i + 1}:"
            )

            print(
                rollout_responses[i]
            )

        print(
            "------------------------------\n"
        )

        # ====================================================
        # 15. Your GRPO implementation
        # ====================================================

        loss, metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,

            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),

            max_grad_norm=max_grad_norm,

            reward_fn=reward_fn,

            repeated_prompts=(
                repeated_prompts
            ),

            rollout_responses=(
                rollout_responses
            ),

            repeated_ground_truths=(
                repeated_ground_truths
            ),

            group_size=group_size,

            baseline="mean",

            advantage_eps=1e-6,

            advantage_normalizer="std",

            importance_reweighting_method="none",

            old_log_probs=None,

            cliprange=None,

            loss_normalization="sequence",

            normalization_constant=None,
        )

        # ====================================================
        # 16. Logging
        # ====================================================

        loss_value = loss.item()

        epoch_loss += loss_value
        num_steps += 1
        global_step += 1


        print(
            f"step={global_step:4d} "
            f"loss={loss_value:.6f} "
            f"reward="
            f"{metadata['mean_reward']:.4f} "
            f"reward_std="
            f"{metadata['std_reward']:.4f} "
            f"reward_min="
            f"{metadata['min_reward']:.4f} "
            f"reward_max="
            f"{metadata['max_reward']:.4f} "
            f"grad_norm="
            f"{metadata['gradient_norm'].item():.4f}"
        )

        wandb.log(
            {
                "train/loss": loss_value,

                "train/mean_reward":
                    metadata["mean_reward"],

                "train/std_reward":
                    metadata["std_reward"],

                "train/min_reward":
                    metadata["min_reward"],

                "train/max_reward":
                    metadata["max_reward"],

                "train/gradient_norm":
                    metadata[
                        "gradient_norm"
                    ].item(),

                "train/mean_format_reward":
                    metadata[
                        "mean_format_reward"
                    ],

                "train/epoch":
                    epoch,

                "train/global_step":
                    global_step,
            }
        )

        # ====================================================
        # 17. 清理 generation cache
        # ====================================================

        del generated
        del prompt_inputs
        del prompt_input_ids
        del prompt_attention_mask

        torch.cuda.empty_cache()

    # ========================================================
    # 18. Epoch statistics
    # ========================================================

    if num_steps > 0:

        mean_epoch_loss = (
            epoch_loss / num_steps
        )

        print(
            f"\nEpoch {epoch + 1} finished. "
            f"mean loss = "
            f"{mean_epoch_loss:.6f}\n"
        )


# ============================================================
# 19. Save model
# ============================================================

output_dir = (
    f"./my_grpo_"
    f"{model_name.replace('/', '-')}"
    f"_gsm8k_final"
)


print(
    f"Saving model to {output_dir}"
)

model.save_pretrained(
    output_dir
)

tokenizer.save_pretrained(
    output_dir
)

print(
    "Training complete!"
)


wandb.finish()