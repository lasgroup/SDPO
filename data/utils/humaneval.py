from datasets import load_dataset, Dataset
import json

from data.format.prompts import CODE_PROMPT

TIME_LIMIT = 5


def _prompt(prefix: str) -> str:
    return CODE_PROMPT.format(problem=f"Your task is to complete the following function. You are not allowed to modify the given code and should do the completion only. Here is the given code to complete: ```python\n{prefix}\n```")


def _parse_description(problem: str, fn_name: str) -> str:
    text = problem.split(f"def {fn_name}")[1]
    if '"""' in text:
        text = text.split('"""')[1]
    elif "'''" in text:
        text = text.split("'''")[1]
    else:
        raise ValueError(f"Unexpected format: {problem}")
    text = text.split(">>>")[0].strip()
    return text


def load_humanevalplus() -> Dataset:
    ds = load_dataset("evalplus/humanevalplus", split="test")
    def format_prompt(ex):
        tests = {
            "inputs": [ex["test"] + "\n" + f'check({ex["entry_point"]})'],
            "outputs": [""],
            "testtype": "code",
            "fn_name": "",
            "time_limit": TIME_LIMIT,
        }
        return {
            "kind": "code",
            "dataset": "humanevalplus",
            "description": _parse_description(ex["prompt"], fn_name=ex["entry_point"]),
            "problem": ex["prompt"],
            "prompt": _prompt(ex["prompt"]),
            "tests": json.dumps(tests, ensure_ascii=False),
        }
    return ds.map(format_prompt, remove_columns=ds.column_names)
