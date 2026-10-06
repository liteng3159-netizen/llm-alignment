import json
from pathlib import Path

from vllm import LLM, SamplingParams

from llm_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "allenai/OLMo-2-0425-1B"

DATA_PATH = Path("data/gsm8k/test.jsonl")

QUESTION_ONLY_PROMPT_PATH = Path(
    "cs336_alignment/prompts/question_only.prompt"
)

R1_ZERO_PROMPT_PATH = Path(
    "cs336_alignment/prompts/r1_zero.prompt"
)

R1_ZERO_THREE_SHOT_PROMPT_PATH = Path(
    "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
)

OUTPUT_PATH = Path(
    "outputs/prompting_baselines.json"
)


# ============================================================
# Load GSM8K
# ============================================================

def load_gsm8k(path: Path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            examples.append(example)

    return examples


# ============================================================
# Load prompt template
# ============================================================

def load_prompt_template(path: Path):
    return path.read_text(encoding="utf-8")


# ============================================================
# Build prompts
# ============================================================

def build_prompts(examples, prompt_template):
    prompts = []

    for example in examples:
        prompt = prompt_template.format(
            question=example["question"]
        )

        prompts.append(prompt)

    return prompts


# ============================================================
# Get GSM8K ground truth
# ============================================================

def get_ground_truth(example):
    """
    GSM8K answer looks like:

    ... reasoning ...

    #### 18

    We only want:
        18
    """

    return example["answer"].split("####")[-1].strip()


# ============================================================
# Categorize result
# ============================================================

def categorize(format_reward, answer_reward):

    if format_reward == 1.0 and answer_reward == 1.0:
        return 1

    if format_reward == 1.0 and answer_reward == 0.0:
        return 2

    if format_reward == 0.0 and answer_reward == 0.0:
        return 3

    raise ValueError(
        f"Unexpected reward combination: "
        f"format={format_reward}, "
        f"answer={answer_reward}"
    )


# ============================================================
# Evaluate one prompt type
# ============================================================

def evaluate_prompt(
    llm,
    examples,
    prompt_name,
    prompt_template,
    reward_fn,
    sampling_params,
):

    # --------------------------------------------------------
    # Build prompts
    # --------------------------------------------------------

    prompts = build_prompts(
        examples,
        prompt_template,
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    results = []

    # --------------------------------------------------------
    # Grade every generation
    # --------------------------------------------------------

    for example, prompt, output in zip(
        examples,
        prompts,
        outputs,
    ):

        response = output.outputs[0].text

        ground_truth = get_ground_truth(example)

        reward = reward_fn(
            response,
            ground_truth,
        )

        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        total_reward = reward["reward"]

        category = categorize(
            format_reward,
            answer_reward,
        )

        results.append(
            {
                "prompt_name": prompt_name,

                "question": example["question"],

                "ground_truth": ground_truth,

                "prompt": prompt,

                "response": response,

                "format_reward": format_reward,

                "answer_reward": answer_reward,

                "reward": total_reward,

                "category": category,
            }
        )

    return results


# ============================================================
# Compute metrics
# ============================================================

def compute_metrics(results):

    category_1 = 0
    category_2 = 0
    category_3 = 0

    for result in results:

        if result["category"] == 1:
            category_1 += 1

        elif result["category"] == 2:
            category_2 += 1

        elif result["category"] == 3:
            category_3 += 1

    total = len(results)

    return {
        "num_examples": total,

        "category_1": category_1,

        "category_2": category_2,

        "category_3": category_3,

        "category_1_rate": (
            category_1 / total
            if total > 0
            else 0.0
        ),

        "category_2_rate": (
            category_2 / total
            if total > 0
            else 0.0
        ),

        "category_3_rate": (
            category_3 / total
            if total > 0
            else 0.0
        ),
    }


# ============================================================
# Print metrics
# ============================================================

def print_metrics(prompt_name, results):

    metrics = compute_metrics(results)

    print()
    print("=" * 80)
    print(prompt_name)
    print("=" * 80)

    print(
        f"Total examples : "
        f"{metrics['num_examples']}"
    )

    print(
        f"Category 1     : "
        f"{metrics['category_1']} "
        f"({metrics['category_1_rate']:.2%})"
    )

    print(
        f"Category 2     : "
        f"{metrics['category_2']} "
        f"({metrics['category_2_rate']:.2%})"
    )

    print(
        f"Category 3     : "
        f"{metrics['category_3']} "
        f"({metrics['category_3_rate']:.2%})"
    )


# ============================================================
# Print examples for manual inspection
# ============================================================

def print_category_examples(
    results,
    category,
    num_examples=10,
):

    selected = [
        result
        for result in results
        if result["category"] == category
    ]

    print()
    print("#" * 80)
    print(
        f"Category {category} "
        f"(showing {min(num_examples, len(selected))})"
    )
    print("#" * 80)

    for i, result in enumerate(
        selected[:num_examples]
    ):

        print()
        print("-" * 80)
        print(f"Example {i + 1}")
        print("-" * 80)

        print("\nQUESTION:")
        print(result["question"])

        print("\nPROMPT:")
        print(result["prompt"])

        print("\nMODEL RESPONSE:")
        print(result["response"])

        print("\nGROUND TRUTH:")
        print(result["ground_truth"])

        print("\nFORMAT REWARD:")
        print(result["format_reward"])

        print("\nANSWER REWARD:")
        print(result["answer_reward"])

        print("\nCATEGORY:")
        print(result["category"])


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    examples = load_gsm8k(
        DATA_PATH
    )

    print(
        f"Loaded {len(examples)} GSM8K examples."
    )

    # --------------------------------------------------------
    # 2. Load prompt templates
    # --------------------------------------------------------

    question_only_template = (
        load_prompt_template(
            QUESTION_ONLY_PROMPT_PATH
        )
    )

    r1_zero_template = (
        load_prompt_template(
            R1_ZERO_PROMPT_PATH
        )
    )

    r1_zero_three_shot_template = (
        load_prompt_template(
            R1_ZERO_THREE_SHOT_PROMPT_PATH
        )
    )

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    llm = LLM(
        model=MODEL_NAME,
    )

    # --------------------------------------------------------
    # 4. Sampling parameters
    # --------------------------------------------------------

    question_only_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,
    )

    r1_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # --------------------------------------------------------
    # 5. question_only
    # --------------------------------------------------------

    question_only_results = evaluate_prompt(
        llm=llm,
        examples=examples,
        prompt_name="question_only",
        prompt_template=question_only_template,
        reward_fn=question_only_reward_fn,
        sampling_params=question_only_sampling_params,
    )

    print_metrics(
        "question_only",
        question_only_results,
    )

    # --------------------------------------------------------
    # 6. r1_zero
    # --------------------------------------------------------

    r1_zero_results = evaluate_prompt(
        llm=llm,
        examples=examples,
        prompt_name="r1_zero",
        prompt_template=r1_zero_template,
        reward_fn=r1_zero_reward_fn,
        sampling_params=r1_sampling_params,
    )

    print_metrics(
        "r1_zero",
        r1_zero_results,
    )

    # --------------------------------------------------------
    # 7. r1_zero_three_shot
    # --------------------------------------------------------

    r1_zero_three_shot_results = evaluate_prompt(
        llm=llm,
        examples=examples,
        prompt_name="r1_zero_three_shot",
        prompt_template=r1_zero_three_shot_template,
        reward_fn=r1_zero_reward_fn,
        sampling_params=r1_sampling_params,
    )

    print_metrics(
        "r1_zero_three_shot",
        r1_zero_three_shot_results,
    )

    # --------------------------------------------------------
    # 8. Final summary table
    # --------------------------------------------------------

    all_results = {
        "question_only": question_only_results,
        "r1_zero": r1_zero_results,
        "r1_zero_three_shot": r1_zero_three_shot_results,
    }

    print()
    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(
        f"{'Prompt':<25}"
        f"{'Category 1':<15}"
        f"{'Category 2':<15}"
        f"{'Category 3':<15}"
    )

    print("-" * 90)

    for prompt_name, results in all_results.items():

        metrics = compute_metrics(results)

        print(
            f"{prompt_name:<25}"
            f"{metrics['category_1']:<15}"
            f"{metrics['category_2']:<15}"
            f"{metrics['category_3']:<15}"
        )

    # --------------------------------------------------------
    # 9. Print examples
    # --------------------------------------------------------

    for prompt_name, results in all_results.items():

        print()
        print()
        print("=" * 90)
        print(prompt_name)
        print("=" * 90)

        # Category 2 is especially important for the assignment.
        print_category_examples(
            results,
            category=2,
            num_examples=10,
        )

        # Category 3 is also important.
        print_category_examples(
            results,
            category=3,
            num_examples=10,
        )

    # --------------------------------------------------------
    # 10. Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "model": MODEL_NAME,

        "metrics": {
            prompt_name: compute_metrics(results)
            for prompt_name, results
            in all_results.items()
        },

        "results": all_results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Results saved to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()