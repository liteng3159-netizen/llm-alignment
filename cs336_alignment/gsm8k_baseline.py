from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
    question_only_reward_fn,
)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "gsm8k" / "test.jsonl"

PROMPT_DIR = PROJECT_ROOT / "cs336_alignment" / "prompts"

PROMPT_FILES = {
    "question_only": PROMPT_DIR / "question_only.prompt",
    "r1_zero": PROMPT_DIR / "r1_zero.prompt",
    "r1_zero_three_shot": PROMPT_DIR / "r1_zero_three_shot_gsm8k.prompt",
}


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

def load_gsm8k(path: Path) -> list[dict[str, str]]:
    """Load GSM8K JSONL and extract each question and final answer."""
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            question = item["question"]
            answer_text = item["answer"]

            # GSM8K answer format:
            # rationale #### final_answer
            if "####" not in answer_text:
                raise ValueError(
                    f"Could not find '####' in GSM8K answer: {answer_text}"
                )

            ground_truth = answer_text.split("####")[-1].strip()

            examples.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                }
            )

    return examples


# ------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------

def load_prompt_template(prompt_type: str) -> str:
    """Load the prompt template corresponding to prompt_type."""
    prompt_path = PROMPT_FILES[prompt_type]

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            "Please check PROMPT_FILES at the top of this script."
        )

    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, question: str) -> str:
    """
    Insert the question into a prompt template.

    Supports common placeholder spellings. If your prompt file uses
    a different placeholder, update this function accordingly.
    """
    placeholders = [
        "{{question}}",
        "{question}",
        "$question",
        "<question>",
    ]

    for placeholder in placeholders:
        if placeholder in template:
            return template.replace(placeholder, question)

    raise ValueError(
        "Could not find a question placeholder in the prompt template. "
        "Expected one of: {{question}}, {question}, $question, <question>."
    )


# ------------------------------------------------------------
# Grading
# ------------------------------------------------------------

def _get_reward_value(
    reward: dict[str, Any],
    possible_names: tuple[str, ...],
) -> float:
    """Read a reward value while allowing a few common key spellings."""
    for name in possible_names:
        if name in reward:
            return float(reward[name])

    raise KeyError(
        f"Could not find any of {possible_names} in reward result: {reward}"
    )


def grade_one(
    prompt_type: str,
    response: str,
    ground_truth: str,
) -> dict[str, float]:
    """
    Grade one model response.

    Assumes the grader returns a dictionary containing format,
    answer/correctness, and total rewards.

    If the grader in your repository uses different keys or argument
    names, modify this function.
    """
    if prompt_type == "question_only":
        reward = question_only_reward_fn(response, ground_truth)
    else:
        reward = r1_zero_reward_fn(response, ground_truth)

    format_reward = _get_reward_value(
        reward,
        ("format_reward", "format"),
    )

    answer_reward = _get_reward_value(
        reward,
        ("answer_reward", "correctness_reward", "correct_reward", "answer"),
    )

    total_reward = _get_reward_value(
        reward,
        ("total_reward", "reward", "total"),
    )

    return {
        "format_reward": format_reward,
        "answer_reward": answer_reward,
        "total_reward": total_reward,
    }


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

def evaluate_prompt(
    server: VLLMServer,
    examples: list[dict[str, str]],
    prompt_type: str,
    batch_size: int,
    max_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    """Generate and grade responses for one prompt type."""

    template = load_prompt_template(prompt_type)

    prompts = [
        build_prompt(template, example["question"])
        for example in examples
    ]

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }

    # Only use the answer-tag stop string for r1_zero-style prompts.
    if prompt_type in ("r1_zero", "r1_zero_three_shot"):
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    if len(completions) != len(examples):
        raise RuntimeError(
            f"Expected {len(examples)} completions, "
            f"but received {len(completions)}."
        )

    records = []
    category_counts = Counter()

    for example, prompt, completion in zip(
        examples, prompts, completions
    ):
        response = completion.text

        rewards = grade_one(
            prompt_type=prompt_type,
            response=response,
            ground_truth=example["ground_truth"],
        )

        format_reward = rewards["format_reward"]
        answer_reward = rewards["answer_reward"]

        # The three categories requested in the assignment.
        if format_reward == 1 and answer_reward == 1:
            category = 1
        elif format_reward == 1 and answer_reward == 0:
            category = 2
        elif format_reward == 0 and answer_reward == 0:
            category = 3
        else:
            category = "other"

        category_counts[category] += 1

        records.append(
            {
                "prompt_type": prompt_type,
                "question": example["question"],
                "ground_truth": example["ground_truth"],
                "prompt": prompt,
                "response": response,
                "token_ids": completion.token_ids,
                "finish_reason": completion.finish_reason,
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "total_reward": rewards["total_reward"],
                "category": category,
            }
        )

    # Save every prompt, generation, and score.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(records)

    summary = {
        "prompt_type": prompt_type,
        "num_examples": n,
        "category_1_count": category_counts[1],
        "category_2_count": category_counts[2],
        "category_3_count": category_counts[3],
        "other_count": category_counts["other"],
        "category_1_rate": category_counts[1] / n if n else 0.0,
        "category_2_rate": category_counts[2] / n if n else 0.0,
        "category_3_rate": category_counts[3] / n if n else 0.0,
        "mean_total_reward": (
            sum(r["total_reward"] for r in records) / n if n else 0.0
        ),
        "output_file": str(output_path),
    }

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo-2-0425-1B on GSM8K prompting baselines."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="Model ID or local model path.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to GSM8K JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gsm8k_baselines",
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens.",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index used by vLLM.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory vLLM may use.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to evaluate (useful for debugging).",
    )

    args = parser.parse_args()

    examples = load_gsm8k(args.data)

    if args.limit is not None:
        examples = examples[:args.limit]

    print(f"Loaded {len(examples)} GSM8K examples.")

    server = VLLMServer(
        model_id=args.model,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    server.start()

    prompt_types = [
        "question_only",
        "r1_zero",
        "r1_zero_three_shot",
    ]

    summaries = []

    for prompt_type in prompt_types:
        print(f"\nEvaluating prompt: {prompt_type}")

        output_path = args.output_dir / f"{prompt_type}.jsonl"

        summary = evaluate_prompt(
            server=server,
            examples=examples,
            prompt_type=prompt_type,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            output_path=output_path,
        )

        summaries.append(summary)

        print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = args.output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()