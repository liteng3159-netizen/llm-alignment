import torch 
from transformers import  PreTrainedModel

def get_response_log_prob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False, 
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)


    # 3. Get the log probability of each target token
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": token_log_probs,
    }

    # 4. Optionally calculate per-token entropy
    if return_token_entropy:
        probs = torch.exp(log_probs)

        token_entropy = -(probs * log_probs).sum(dim=-1)

        result["token_entropy"] = token_entropy

    return result





