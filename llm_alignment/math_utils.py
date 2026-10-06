import re


def extract_answer(text):

    # ========================================================
    # 1. 优先解析 <answer>...</answer>
    # ========================================================

    answer_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        re.DOTALL,
    )

    if answer_match:

        answer_text = answer_match.group(1)

        numbers = re.findall(
            r"\d+",
            answer_text,
        )

        if numbers:
            return int(numbers[-1])

    # ========================================================
    # 2. 兼容 GSM8K #### 格式
    # ========================================================

    if "####" in text:

        idx = text.find("####")

        after_hashes = text[
            idx + 4:
        ]

        numbers = re.findall(
            r"\d+",
            after_hashes,
        )

        if numbers:
            return int(numbers[0])

    # ========================================================
    # 3. fallback
    # ========================================================

    all_numbers = re.findall(
        r"\d+",
        text,
    )

    if all_numbers:
        return int(all_numbers[-1])

    return None