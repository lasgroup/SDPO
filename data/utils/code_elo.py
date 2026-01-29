from datasets import load_dataset, Dataset
import json

from data.format.prompts import CODE_PROMPT
from data.utils.code import float_with_default

TIME_LIMIT = 1


def _make_html_problem(problem, time_limit: float) -> str:
    # We do not include `problem["examples"]` since this enables reward hacking if private test cases are included.

    title = problem["title"]
    html_output = "<html><body>"
    html_output += f"<h1>{title}</h1>"
    html_output += f'<div>Time limit per test: {time_limit} s</div>'
    html_output += f"<h2>Description</h2>"
    html_output += f"<div>{problem['description']}</div>"
    html_output += f"<h2>Input</h2>"
    html_output += f"<div>{problem['input']}</div>"
    html_output += f"<h2>Output</h2>"
    html_output += f"<div>{problem['output']}</div>"

    if problem["interaction"]:
        html_output += f"<h2>Interaction</h2>"
        html_output += f"<div>{problem['interaction']}</div>"
    if problem["note"]:
        html_output += f"<h2>Note</h2>"
        html_output += f"<div>{problem['note']}</div>"
    html_output += "</body></html>"
    return html_output


def load_code_elo() -> Dataset:
    ds = load_dataset("Qwen/CodeElo", split="test")
    ds = ds.rename_columns({"rating": "elo"})

    def format_prompt(ex):
        time_limit = float_with_default(ex["time_limit_ms"] / 1000, TIME_LIMIT)
        problem = _make_html_problem(ex, time_limit=time_limit)
        tests = {
            "inputs": [t[0] for t in ex["examples"]],
            "outputs": [t[1] for t in ex["examples"]],
            "testtype": "stdin",
            "fn_name": "",
            "time_limit": time_limit,
        }
        return {
            "kind": "code",
            "dataset": "code_elo",
            "description": ex["description"],
            "problem": problem,
            "prompt": CODE_PROMPT.format(problem=problem),
            "tests": json.dumps(tests, ensure_ascii=False)
        }
    return ds.map(format_prompt, remove_columns=ds.column_names)
