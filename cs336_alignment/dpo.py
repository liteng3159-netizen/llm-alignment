from pathlib import Path

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_lm,
    reference_lm,
    tokenizer,
    beta,
    prompt,
    chosen_response,
    rejected_response,
):
    # =========================================================
    # 1. Load Alpaca template
    # =========================================================
    template_path = (
        Path(__file__).parent
        / "prompts_safety"
        / "alpaca_sft.prompt"
    )

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # =========================================================
    # 2. Build prompt-only text
    # =========================================================
    prompt_text = template.format(
        instruction=prompt,
        response="",
    )

    # =========================================================
    # 3. Build chosen / rejected text
    # =========================================================
    chosen_text = template.format(
        instruction=prompt,
        response=chosen_response,
    )

    rejected_text = template.format(
        instruction=prompt,
        response=rejected_response,
    )

    # =========================================================
    # 4. Tokenize
    # =========================================================
    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    chosen_ids = tokenizer(
        chosen_text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    rejected_ids = tokenizer(
        rejected_text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    # =========================================================
    # 5. Append EOS
    # =========================================================
    eos = torch.tensor(
        [[tokenizer.eos_token_id]],
        dtype=torch.long,
    )

    chosen_ids = torch.cat(
        [chosen_ids, eos],
        dim=1,
    )

    rejected_ids = torch.cat(
        [rejected_ids, eos],
        dim=1,
    )

    # =========================================================
    # 6. Response masks
    # =========================================================
    chosen_response_start = prompt_ids.shape[1]

    rejected_response_start = prompt_ids.shape[1]

    chosen_mask = torch.zeros_like(chosen_ids)

    rejected_mask = torch.zeros_like(rejected_ids)

    chosen_mask[:, chosen_response_start:] = 1
    rejected_mask[:, rejected_response_start:] = 1

    # =========================================================
    # 7. Get devices
    # =========================================================
    policy_device = next(
        policy_lm.parameters()
    ).device

    reference_device = next(
        reference_lm.parameters()
    ).device

    # =========================================================
    # 8. Compute response log probability
    # =========================================================
    def get_log_prob(
        model,
        input_ids,
        response_mask,
        device,
    ):
        input_ids = input_ids.to(device)
        response_mask = response_mask.to(device)

        logits = model(input_ids).logits

        # causal LM shift
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]

        # [B, T-1, V]
        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        # [B, T-1]
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        # mask needs the same causal shift
        response_mask = response_mask[:, 1:]

        # only response tokens
        response_log_probs = (
            token_log_probs * response_mask
        )

        # sum response token log probabilities
        return response_log_probs.sum(dim=-1)

    # =========================================================
    # 9. Policy log probabilities
    # =========================================================
    policy_chosen = get_log_prob(
        policy_lm,
        chosen_ids,
        chosen_mask,
        policy_device,
    )

    policy_rejected = get_log_prob(
        policy_lm,
        rejected_ids,
        rejected_mask,
        policy_device,
    )

    # =========================================================
    # 10. Reference log probabilities
    # =========================================================
    with torch.no_grad():

        reference_chosen = get_log_prob(
            reference_lm,
            chosen_ids,
            chosen_mask,
            reference_device,
        )

        reference_rejected = get_log_prob(
            reference_lm,
            rejected_ids,
            rejected_mask,
            reference_device,
        )

    # =========================================================
    # 11. DPO objective
    # =========================================================
    logits = beta * (
        policy_chosen
        - policy_rejected
        - reference_chosen.to(policy_device)
        + reference_rejected.to(policy_device)
    )

    loss = -F.logsigmoid(logits)

    return loss.mean()