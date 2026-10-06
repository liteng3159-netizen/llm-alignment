import torch
from transformers import PreTrainedTokenizer

#字符串->token ID

# 把 prompt 和模型生成的 response 拼起来，然后构造训练语言模型需要的 input_ids、labels 和 response_mask。

def tokenize_prompt_and_output(
    prompt_strs = list[str],
    output_strs = list[str],
    tokenizer = PreTrainedTokenizer,

) -> dict[str, torch.Tensor]:

    """
    最后的返回值
        {
        "input_ids": ...,
        "labels": ...,
        "response_mask": ...
        }
    """

    # Tokenize prompt and output separately
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
    )

    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
    )

    # Concatenate prompt and output
    prompt_and_output = [
        prompt + output
        for prompt, output in zip(
            prompt_tokens["input_ids"],
            output_tokens["input_ids"],
        )
    ]

    response_mask = [
        [0] * len(prompt) + [1] * len(output)
        for prompt, output in zip(
            prompt_tokens["input_ids"],
            output_tokens["input_ids"],
        )
    ]

    # Pad input IDs
    padded = tokenizer.pad(
        {"input_ids": prompt_and_output},
        padding=True,
        return_tensors="pt",
    )

    # Pad response mask
    response_mask = tokenizer.pad(
        {"input_ids": response_mask},
        padding=True,
        return_tensors="pt",
    )["input_ids"]

    # input_ids: remove the final token
    input_ids = padded["input_ids"][:, :-1]

    # labels: remove the first token
    labels = padded["input_ids"][:, 1:]

    # Align response mask with labels
    response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

