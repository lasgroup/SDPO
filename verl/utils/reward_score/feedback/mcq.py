import re


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def is_correct_format(text: str) -> bool:
    """
    Check if the text is in the correct XML format.

    The text should contain at the end of the text:
    <answer>
    (A|B|C|D)
    </answer>
    """
    pattern = r"<answer>\s*(A|B|C|D)\s*</answer>$"
    return re.search(pattern, text) is not None

def compute_score(solution: str, ground_truth: str) -> dict:
    multiple_choice_answer = extract_xml_answer(solution)

    reward = float(multiple_choice_answer == ground_truth)
    incorrect_format = is_correct_format(solution)

    return {
      "score": reward,
      "acc": reward,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if incorrect_format else 0,
      "feedback": "",
    }
