import torch
from typing import Callable
from typing import Literal
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer

#给每个模型的回答打分 reward
def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:

    rewards = []
    format_rewards = []

    for response, ground_truth in zip(
        rollout_responses,
        repeated_ground_truths,
    ):
        reward_info = reward_fn(response, ground_truth)

        rewards.append(reward_info["reward"])
        format_rewards.append(reward_info["format_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)

    metadata = {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
    }

    return raw_rewards, metadata

#(reward - group_mean) / group_std
# group_size是一个prompt生成多少个回答 最后返回的就是一个得分
def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,  
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):

    if baseline != "mean":
        raise NotImplementedError(
            f"Unsupported baseline: {baseline}"
        )

    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage_normalizer: {advantage_normalizer}"
        )
    
    rewards = raw_rewards.reshape(-1, group_size)

    # 每组的平均 reward
    group_mean = rewards.mean(dim=1, keepdim=True)

    # baseline = mean
    advantages = rewards - group_mean

    #每组的平均标准差 
    group_std = rewards.std(dim=1, keepdim=True)

    # 标准化
    advantages = advantages / (group_std + advantage_eps)

    # 恢复成 [rollout_batch_size] 就是变成一维
    advantages = advantages.reshape(-1)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }

    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if importance_reweighting_method != "none":
        raise NotImplementedError(
            f"Unsupported importance_reweighting_method: "
            f"{importance_reweighting_method}"
        )
    
    advantages = raw_rewards_or_advantages.reshape(-1, 1)


    per_token_policy_gradient_loss = -advantages * policy_log_probs

    metadata = {}

    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:

    if loss_normalization != "sequence":
        raise NotImplementedError

    masked_loss = per_token_policy_gradient_loss * mask

    loss_per_sequence = (masked_loss.sum(dim = -1) / mask.sum(dim = -1))

    return loss_per_sequence.mean()

#完成一次参数更新
def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:

    # ============================================================
    # 1. 只支持题目要求的 standard on-policy GRPO
    # ============================================================

    if baseline != "mean":
        raise NotImplementedError(
            f"Unsupported baseline: {baseline}"
        )

    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage_normalizer: "
            f"{advantage_normalizer}"
        )

    if importance_reweighting_method != "none":
        raise NotImplementedError(
            f"Unsupported importance_reweighting_method: "
            f"{importance_reweighting_method}"
        )

    if loss_normalization != "sequence":
        raise NotImplementedError(
            f"Unsupported loss_normalization: "
            f"{loss_normalization}"
        )

    if old_log_probs is not None:
        raise NotImplementedError(
            "old_log_probs is only needed for importance "
            "reweighting."
        )

    if cliprange is not None:
        raise NotImplementedError(
            "cliprange is only needed for importance "
            "reweighting."
        )

    # ============================================================
    # 2. 基本检查
    # ============================================================

    batch_size = len(rollout_responses)

    if len(repeated_prompts) != batch_size:
        raise ValueError(
            "repeated_prompts and rollout_responses "
            "must have the same length."
        )

    if len(repeated_ground_truths) != batch_size:
        raise ValueError(
            "repeated_ground_truths and rollout_responses "
            "must have the same length."
        )

    if batch_size % group_size != 0:
        raise ValueError(
            "batch_size must be divisible by group_size."
        )

    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "batch_size must be divisible by "
            "gradient_accumulation_steps."
        )

    # ============================================================
    # 3. 计算 rewards
    #    使用你已经写好的 compute_rollout_rewards
    # ============================================================

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    # ============================================================
    # 4. 计算 group-normalized advantages
    #    使用你已经写好的 helper
    # ============================================================

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
        )
    )

    # ============================================================
    # 5. 确定 device
    # ============================================================

    device = next(model.parameters()).device

    advantages = advantages.to(device)

    # ============================================================
    # 6. 分别 tokenize prompt 和 response
    #
    # 不要 tokenizer(prompt + response)
    # ============================================================

    prompt_tokenized = tokenizer(
        repeated_prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    response_tokenized = tokenizer(
        rollout_responses,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    prompt_ids = prompt_tokenized["input_ids"]
    response_ids = response_tokenized["input_ids"]

    # ============================================================
    # 7. 拼接 prompt + response，并构造 response mask
    # ============================================================

    sequences = []
    response_masks = []

    for p_ids, r_ids in zip(
        prompt_ids,
        response_ids,
    ):
        sequence = p_ids + r_ids

        response_mask = (
            [0] * len(p_ids)
            + [1] * len(r_ids)
        )

        sequences.append(sequence)
        response_masks.append(response_mask)

    # ============================================================
    # 8. Padding
    # ============================================================

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        raise ValueError(
            "tokenizer.pad_token_id must not be None."
        )

    max_seq_len = max(
        len(sequence)
        for sequence in sequences
    )

    input_ids = torch.full(
        (batch_size, max_seq_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.long,
        device=device,
    )

    response_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.float32,
        device=device,
    )

    for i, (
        sequence,
        sequence_response_mask,
    ) in enumerate(
        zip(sequences, response_masks)
    ):
        seq_len = len(sequence)

        input_ids[i, :seq_len] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )

        attention_mask[i, :seq_len] = 1

        response_mask[i, :seq_len] = torch.tensor(
            sequence_response_mask,
            dtype=torch.float32,
            device=device,
        )

    # ============================================================
    # 9. microbatch
    # ============================================================

    microbatch_size = (
        batch_size // gradient_accumulation_steps
    )

    optimizer.zero_grad()

    total_loss = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )

    # ============================================================
    # 10. forward + backward
    # ============================================================

    for microbatch_idx in range(
        gradient_accumulation_steps
    ):

        start = (
            microbatch_idx
            * microbatch_size
        )

        end = start + microbatch_size

        micro_input_ids = input_ids[
            start:end
        ]

        micro_attention_mask = attention_mask[
            start:end
        ]

        micro_response_mask = response_mask[
            start:end
        ]

        micro_advantages = advantages[
            start:end
        ]

        # --------------------------------------------------------
        # 10.1 Forward
        # --------------------------------------------------------

        outputs = model(
            input_ids=micro_input_ids,
            attention_mask=micro_attention_mask,
        )

        logits = outputs.logits

        # --------------------------------------------------------
        # 10.2 Shift
        # --------------------------------------------------------

        shift_logits = logits[:, :-1, :]
        shift_labels = micro_input_ids[:, 1:]

        shift_response_mask = (
            micro_response_mask[:, 1:]
        )

        # --------------------------------------------------------
        # 10.3 log probability
        # --------------------------------------------------------

        log_probs = torch.log_softmax(
            shift_logits,
            dim=-1,
        )

        policy_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        # --------------------------------------------------------
        # 10.4 policy gradient loss
        #     使用你已有的 helper
        # --------------------------------------------------------

        per_token_loss, loss_metadata = (
            compute_policy_gradient_loss(
                raw_rewards_or_advantages=micro_advantages,
                policy_log_probs=policy_log_probs,
                importance_reweighting_method="none",
                old_log_probs=None,
                cliprange=None,
                response_mask=shift_response_mask,
            )
        )

        # --------------------------------------------------------
        # 10.5 aggregate
        #     使用你已有的 helper
        # --------------------------------------------------------

        microbatch_loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=per_token_loss,
                mask=shift_response_mask,
                loss_normalization="sequence",
                normalization_constant=None,
            )
        )

        # --------------------------------------------------------
        # 10.6 gradient accumulation
        # --------------------------------------------------------

        scaled_loss = (
            microbatch_loss
            / gradient_accumulation_steps
        )

        scaled_loss.backward()

        total_loss = (
            total_loss
            + microbatch_loss.detach()
            / gradient_accumulation_steps
        )

    # ============================================================
    # 11. gradient clipping
    # ============================================================

    if max_grad_norm is not None:

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

        grad_norm_value = (
            grad_norm.detach()
        )

    else:

        grad_norm_squared = torch.zeros(
            (),
            dtype=torch.float32,
            device=device,
        )

        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            grad_norm_squared += (
                parameter.grad.detach()
                .pow(2)
                .sum()
            )

        grad_norm_value = (
            grad_norm_squared.sqrt()
        )

    # ============================================================
    # 12. optimizer step
    # ============================================================

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # ============================================================
    # 13. metadata
    # ============================================================

    metadata = {
        **reward_metadata,
        **advantage_metadata,
        "gradient_norm": grad_norm_value,
    }

    return total_loss.detach(), metadata