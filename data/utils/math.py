import re


def process_gsm8k(ex):
    full_solution = ex["answer"]
    # extract integer or float after '####'
    m = re.search(r"####\s*([0-9]+(?:\.[0-9]+)?)", full_solution)
    ans_str = m.group(1) if m else ""
    return {
        "solution": full_solution,
        "answer": ans_str,
    }
