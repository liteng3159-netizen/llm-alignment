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
    # 2. Format chosen / rejected
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
    # 3. Tokenize
    # =========================================================
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
    # 4. Append EOS
    # =========================================================
    eos = torch.tensor(
        [[tokenizer.eos_token_id]],
        dtype=torch.long,
    )

    chosen_ids = torch.cat([chosen_ids, eos], dim=1)
    rejected_ids = torch.cat([rejected_ids, eos], dim=1)

    # =========================================================
    # 5. Get devices
    # =========================================================
    policy_device = next(policy_lm.parameters()).device
    reference_device = next(reference_lm.parameters()).device

    # =========================================================
    # 6. Compute log probabilities
    # =========================================================
    def get_log_prob(model, input_ids, device):
        input_ids = input_ids.to(device)

        logits = model(input_ids).logits

        # causal LM shift
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        return token_log_probs.sum(dim=-1)

    # policy model
    policy_chosen = get_log_prob(
        policy_lm,
        chosen_ids,
        policy_device,
    )

    policy_rejected = get_log_prob(
        policy_lm,
        rejected_ids,
        policy_device,
    )

    # reference model
    with torch.no_grad():
        reference_chosen = get_log_prob(
            reference_lm,
            chosen_ids,
            reference_device,
        )

        reference_rejected = get_log_prob(
            reference_lm,
            rejected_ids,
            reference_device,
        )
    
    # =========================================================
    # 7. DPO loss
    # 
    # =========================================================
    logits = beta * (
        policy_chosen
        - policy_rejected
        - reference_chosen.to(policy_device)
        + reference_rejected.to(policy_device)
    )

    loss = -F.logsigmoid(logits)

    return loss.squeeze()