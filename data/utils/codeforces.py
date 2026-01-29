from datasets import load_dataset, Dataset
import json

from data.utils.code import float_with_default

TIME_LIMIT = 1


def _make_html_problem(problem) -> str:
    # We do not include `problem["examples"]` since this enables reward hacking if private test cases are included.

    title = problem["title"]
    html_output = "<html><body>"
    html_output += f"<h1>{title}</h1>"
    html_output += f'<div>Time limit per test: {problem["time_limit"]} s</div>'
    html_output += f"<h2>Description</h2>"
    html_output += f"<div>{problem['description']}</div>"
    html_output += f"<h2>Input Format</h2>"
    html_output += f"<div>{problem['input_format']}</div>"
    html_output += f"<h2>Output Format</h2>"
    html_output += f"<div>{problem['output_format']}</div>"

    if problem["interaction_format"]:
        html_output += f"<h2>Interaction</h2>"
        html_output += f"<div>{problem['interaction_format']}</div>"
    if problem["note"]:
        html_output += f"<h2>Note</h2>"
        html_output += f"<div>{problem['note']}</div>"
    if problem["editorial"]:
        html_output += f"<h2>Editorial</h2>"
        html_output += f"<div>{problem['editorial']}</div>"
    html_output += "</body></html>"
    return html_output


def load_codeforces(dataset_split: str) -> Dataset:
    ds = load_dataset("open-r1/codeforces", "verifiable", split=dataset_split)
    ds = ds.rename_columns({"rating": "elo"})

    def format_prompt(ex):
        assert isinstance(ex["official_tests"], list)
        tests = {
            "inputs": [t["input"] for t in ex["official_tests"]],
            "outputs": [t["output"] for t in ex["official_tests"]],
            "testtype": "stdin",
            "fn_name": "",
            "time_limit": float_with_default(ex["time_limit"], TIME_LIMIT),
        }
        return {
            "kind": "code",
            "dataset": "codeforces",
            "description": ex["description"],
            "elo": ex["elo"],
            "problem": _make_html_problem(ex),
            "tests": json.dumps(tests, ensure_ascii=False)
        }
    return ds.map(format_prompt, remove_columns=ds.column_names)
