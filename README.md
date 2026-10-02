# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

/mnt/data/lt/hf_home/hub/models--allenai--OLMo-2-0425-1B

PYTHONPATH=. .venv/bin/python scripts/train_grpo.py \
    /mnt/data/lt/hf_home/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf


before grpo

| allenai--OLMo-2-0425-1B | 38.59 |

after grpo

| allenai--OLMo-2-0425-1B | 52.92 |



before dpo

| Qwen/Qwen2.5-3B-Instruct | 68.60 |

after dpo



before sft

| Qwen2.5-3B-gsm8k | 73.84 |

after sft

| Qwen2.5-3B-ultrachat-sft-gsm8k | 72.55 |


.venv/bin/python -m pytest -k test_packed_sft_dataset
