import re
from typing import Optional


def extract_final(text: str) -> Optional[str]:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else "" #None


def extract_again(text: str) -> Optional[str]:
    match = re.search(r"Answer:\s*([A-J])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_answer(text: str) -> Optional[str]:
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def compute_score(solution, ground_truth):
    multiple_choice_answer = extract_answer(solution)
    reward = float(multiple_choice_answer == ground_truth)
    incorrect_format = multiple_choice_answer not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    return {"score": reward, "acc": reward, "pred": multiple_choice_answer, "incorrect_format": 1 if incorrect_format else 0}

