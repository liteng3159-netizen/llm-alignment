import re #python中的正则表达式模块

def parse_mmlu_response(
        mmlu_example: dict,
        response: str,
        ) -> str | None:
    """
    parse_mmlu_response 的 Docstring
    
    :param response: 说明
    :type response: str
    :return: 说明
    :rtype: str | None
    这个函数的作用就是找出 A B C D
    """
    response = response.strip().upper()
    if response in {"A", "B", "C", "D"}:
        return response
    
    match = re.search(
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([ABCD])\b",
        response,
    )

    if match:
        return match.group(1)

    # 无法确定答案时返回 None
    return None