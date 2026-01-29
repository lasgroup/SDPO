import re


def get_multiple_choice_answer(pred: str):
    tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")

    return pred


def compute_score(solution, ground_truth):
    multiple_choice_answer = get_multiple_choice_answer(solution)
    reward = float(multiple_choice_answer == ground_truth)
    incorrect_format = multiple_choice_answer not in ["A", "B", "C", "D"]
    return {
      "score": reward,
      "acc": reward,
      "pred": multiple_choice_answer,
      "incorrect_format": 1 if incorrect_format else 0,
      "feedback": "",
    }
