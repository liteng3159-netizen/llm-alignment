import argparse
import math
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.sft_dataset import PackedSFTDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # Model / data
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/mnt/data/lt/hf_home/models/Qwen2.5-3B",
    )
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default="/mnt/data/lt/homework5/data/ultrachat_train.jsonl.gz",
    )
    parser.add_argument(
        "--valid_dataset_path",
        type=str,
        default="/mnt/data/lt/homework5/data/ultrachat_test.jsonl.gz",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/lt/homework5/outputs/ultrachat-sft",
    )

    # Training
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument("--num_epochs", type=int, default=1)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Scheduler
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1000)

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:4",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
    )

    return parser.parse_args()


def compute_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(input_ids)

    logits = outputs.logits

    # logits:
    #   (batch_size, seq_length, vocab_size)
    #
    # labels:
    #   (batch_size, seq_length)
    #
    # cross_entropy 要求：
    #   input  -> (N, vocab_size)
    #   target -> (N,)
    #
    # 所以把 batch 和 sequence 两个维度展平。

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
    )

    return loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids).logits

        # reduction="sum"，方便按照 token 数量计算平均 loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        )

        num_tokens = labels.numel()

        total_loss += loss.item()
        total_tokens += num_tokens

    model.train()

    return total_loss / total_tokens


def get_cosine_lr(
    step,
    total_steps,
    warmup_steps,
    learning_rate,
):
    # Linear warmup
    if step < warmup_steps:
        return learning_rate * step / max(warmup_steps, 1)

    # Cosine decay
    progress = (
        step - warmup_steps
    ) / max(total_steps - warmup_steps, 1)

    progress = min(max(progress, 0.0), 1.0)

    return learning_rate * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Device
    # ---------------------------------------------------------

    device = torch.device(args.device)

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # ---------------------------------------------------------
    # 2. Tokenizer
    # ---------------------------------------------------------

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path
    )

    # ---------------------------------------------------------
    # 3. Dataset
    # ---------------------------------------------------------

    print("Loading datasets...")

    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_dataset_path,
        seq_length=args.seq_length,
        shuffle=False,
    )

    valid_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.valid_dataset_path,
        seq_length=args.seq_length,
        shuffle=False,
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")

    # ---------------------------------------------------------
    # 4. DataLoader
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # 5. Load model
    # ---------------------------------------------------------

    print("Loading model...")

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }

    if args.use_flash_attention:
        model_kwargs["attn_implementation"] = (
            "flash_attention_2"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    model.to(device)

    print("Model loaded.")

    # ---------------------------------------------------------
    # 6. Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---------------------------------------------------------
    # 7. Calculate training steps
    # ---------------------------------------------------------

    num_microbatches_per_epoch = len(train_loader)

    optimizer_steps_per_epoch = math.ceil(
        num_microbatches_per_epoch
        / args.gradient_accumulation_steps
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * args.num_epochs
    )

    warmup_steps = int(
        total_optimizer_steps
        * args.warmup_ratio
    )

    print(
        f"Microbatches per epoch: "
        f"{num_microbatches_per_epoch}"
    )

    print(
        f"Optimizer steps per epoch: "
        f"{optimizer_steps_per_epoch}"
    )

    print(
        f"Total optimizer steps: "
        f"{total_optimizer_steps}"
    )

    print(
        f"Warmup steps: "
        f"{warmup_steps}"
    )

    # ---------------------------------------------------------
    # 8. Training
    # ---------------------------------------------------------

    global_step = 0

    model.train()
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):

        print(
            f"\n========== Epoch {epoch + 1} "
            f"/ {args.num_epochs} =========="
        )

        running_loss = 0.0

        for microbatch_idx, batch in enumerate(
            train_loader
        ):

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            loss = compute_loss(
                model,
                batch,
                device,
            )

            # 保存原始 loss 用于 logging
            running_loss += loss.item()

            # -------------------------------------------------
            # Gradient accumulation
            # -------------------------------------------------

            loss_for_backward = (
                loss
                / args.gradient_accumulation_steps
            )

            loss_for_backward.backward()

            # -------------------------------------------------
            # Optimizer step
            # -------------------------------------------------

            is_last_microbatch = (
                microbatch_idx
                == len(train_loader) - 1
            )

            should_update = (
                (microbatch_idx + 1)
                % args.gradient_accumulation_steps
                == 0
                or is_last_microbatch
            )

            if should_update:

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                # 学习率
                lr = get_cosine_lr(
                    global_step,
                    total_optimizer_steps,
                    warmup_steps,
                    args.learning_rate,
                )

                set_learning_rate(
                    optimizer,
                    lr,
                )

                optimizer.step()

                optimizer.zero_grad()

                global_step += 1

                # -------------------------------------------------
                # Logging
                # -------------------------------------------------

                if (
                    global_step % args.log_every == 0
                    or global_step == 1
                ):
                    avg_loss = (
                        running_loss
                        / args.gradient_accumulation_steps
                    )

                    print(
                        f"step={global_step:5d} "
                        f"train_loss={avg_loss:.4f} "
                        f"lr={lr:.6e} "
                        f"grad_norm={grad_norm.item():.4f}"
                    )

                    running_loss = 0.0

                # -------------------------------------------------
                # Validation
                # -------------------------------------------------

                if (
                    global_step % args.eval_every == 0
                ):
                    val_loss = evaluate(
                        model,
                        valid_loader,
                        device,
                    )

                    print(
                        f"step={global_step:5d} "
                        f"val_loss={val_loss:.4f}"
                    )

    # ---------------------------------------------------------
    # 9. Final validation
    # ---------------------------------------------------------

    final_val_loss = evaluate(
        model,
        valid_loader,
        device,
    )

    print(
        f"\nFinal validation loss: "
        f"{final_val_loss:.4f}"
    )

    # ---------------------------------------------------------
    # 10. Save model + tokenizer
    # ---------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(
        f"Saving model to {args.output_dir}"
    )

    model.save_pretrained(
        save_directory=args.output_dir
    )

    tokenizer.save_pretrained(
        save_directory=args.output_dir
    )

    print("Training finished.")


if __name__ == "__main__":
    main()