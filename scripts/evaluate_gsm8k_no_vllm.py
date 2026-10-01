"""
cd /mnt/data/lt/homework5
PYTHONPATH=. .venv/bin/python scripts/evaluate_gsm8k_no_vllm.py /mnt/data/lt/hf_home/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf
"""
# python evaluate_gsm8k_no_vllm.py /mnt/data/lt/hf_home/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf

import json
import os
import re
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer  #用HF加载模型
import torch
from tqdm import tqdm
from cs336_alignment.math_utils import extract_answer #从回答中解析出答案

# Model to evaluate
model_name = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/gpt-neo-2.7B"

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set pad token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set padding side to left for decoder-only models
tokenizer.padding_side = "left"

# Load evaluation data
eval_data = []
with open("data/gsm8k/test.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        eval_data.append({
            "question": item["question"],
            "numerical_answer": extract_answer(item["answer"]),
        })

PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Assistant: <think> There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6. </think> <answer> 6 </answer>
User: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Assistant: <think> There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5. </think> <answer> 5 </answer>
User: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Assistant: <think> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39. </think> <answer> 39 </answer>
User: {question}
Assistant: <think>"""

# Prepare prompts
prompts = [
    PROMPT_TEMPLATE.format(question=item["question"])
    for item in eval_data
]

# Generate completions
generated_texts = []
print(f"Generating completions for {len(prompts)} problems...")

# Batch processing for efficiency
batch_size = 128  # Adjust based on GPU memory
for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
    batch_prompts = prompts[i:i+batch_size]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    input_width = inputs["input_ids"].shape[1]

    for output in outputs:
        response_ids = output[input_width:]
        generated_text = tokenizer.decode(
            response_ids,
            skip_special_tokens=True
        ).strip()
        generated_texts.append(generated_text)

# Evaluate
correct = 0
incorrect_completions = []

for i, generated_text in enumerate(generated_texts):
    predicted_answer = extract_answer(generated_text)
    ground_truth = eval_data[i]["numerical_answer"]

    if predicted_answer == ground_truth:
        correct += 1
    else:
        incorrect_completions.append({
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth
        })

accuracy = correct / len(eval_data)
print(f"\nAccuracy: {correct}/{len(eval_data)} = {accuracy:.2%}")

# Save incorrect completions
os.makedirs("evals", exist_ok=True)
with open(f"evals/{model_name.replace('/', '-')}_incorrect_completions.json", "w") as f:
    json.dump(incorrect_completions, f, indent=2)
print(f"Saved {len(incorrect_completions)} incorrect completions to evals/{model_name.replace('/', '-')}_incorrect_completions.json")
