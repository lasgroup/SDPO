from datasets import load_dataset, Dataset

from data.format.prompts import PROMPT
from data.utils.math import process_gsm8k

PROBLEM_KEY = {
    "math-ai/aime25": "problem",
    "math-ai/aime24":"problem",
    "math-ai/amc23":"question",
    "math-ai/math500":"problem",
    "openai/gsm8k": "question"
}
ANSWER_KEY = {
    "math-ai/aime25": "answer",
    "math-ai/aime24":"solution",
    "math-ai/amc23":"answer",
    "math-ai/math500":"answer",
    "openai/gsm8k": "answer"
}


def _format_math(ex, dataset_name: str) -> dict:
    return {
        "kind": "math",
        "dataset": dataset_name.split("/")[1],
        "description": ex[PROBLEM_KEY[dataset_name]],
        "problem": ex[PROBLEM_KEY[dataset_name]],
        "prompt": PROMPT.format(problem=ex[PROBLEM_KEY[dataset_name]]),
        "answer": str(ex[ANSWER_KEY[dataset_name]]),
    }


def load_math(dataset_name: str) -> Dataset:
    assert dataset_name in ["math-ai/aime24", "math-ai/aime25", "math-ai/math500", "math-ai/amc23", "openai/gsm8k"]

    if dataset_name == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    else:
        ds = load_dataset(dataset_name, split="test")

    if dataset_name == "math-ai/aime24":  # remove \boxed{}
        ds = ds.map(lambda ex: {ANSWER_KEY[dataset_name]: ex[ANSWER_KEY[dataset_name]][7:-1]}, desc="AIME24 answer extraction")
    if dataset_name == "openai/gsm8k":
        ds = ds.map(process_gsm8k, desc="GSM8K answer extraction")

    return ds.map(lambda ex: _format_math(ex, dataset_name=dataset_name), remove_columns=ds.column_names)
