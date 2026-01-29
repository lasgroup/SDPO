from datasets import load_dataset, Dataset
import json

from data.format.prompts import CODE_PROMPT

TIME_LIMIT = 5

def _prompt(problem: str, fn_name: str) -> str:
    return CODE_PROMPT.format(problem=f'{problem} The function should be called `{fn_name}`.')


def _parse_fn_name(code: str) -> str:
    return code.split(f"def ")[1].split('(')[0].strip()


def load_mbppplus() -> Dataset:
    ds = load_dataset("evalplus/mbppplus", split="test")
    def format_prompt(ex):
        fn_name = _parse_fn_name(ex["code"])
        test = ""
        for imp in ex["test_imports"]:
            test += f'{imp}\n'
        test += ex["test"]
        tests = {
            "inputs": [test],
            "outputs": [""],
            "testtype": "code",
            "fn_name": "",
            "time_limit": TIME_LIMIT,
        }
        return {
            "kind": "code",
            "dataset": "mbppplus",
            "description": ex["prompt"],
            "problem": ex["prompt"],
            "prompt": _prompt(ex["prompt"], fn_name),
            "tests": json.dumps(tests, ensure_ascii=False),
        }
    return ds.map(format_prompt, remove_columns=ds.column_names)
