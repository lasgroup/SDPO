from datasets import concatenate_datasets, load_dataset, Dataset
import json

from data.utils.code import parse_description, float_with_default

TIME_LIMIT = 1


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _parse_testtype(tests):
    if "fn_name" in tests and tests["fn_name"] != "":
        return "functional"
    else:
        return "stdin"


def _translate_test_cases(data, time_limit = None):
    tests = json.loads(data)
    if isinstance(tests, dict):
        assert len(tests["inputs"]) == len(tests["outputs"])
        out_tests = {
            "inputs": tests["inputs"],
            "outputs": tests["outputs"],
            "testtype": _parse_testtype(tests),
            "fn_name": tests.get("fn_name", ""),
            "time_limit": float_with_default(time_limit, TIME_LIMIT),
        }
    else:
        raise ValueError(f"Unexpected type for tests: {type(tests)}")
    return json.dumps(out_tests, ensure_ascii=False)


def load_taco() -> Dataset:
    ds = load_dataset("likaixin/TACO-verified", split="train")

    def format_prompt(ex):
        problem = ex["question"]  # already includes public test cases
        if ex["starter_code"].strip() != "" and "def " in ex["starter_code"]:
            problem += f"\n\nYour solution should have the following signature: ```python\n{_parse_signature(ex['starter_code'])}\n```"
        description = parse_description(problem)

        return {
            "kind": "code",
            "dataset": "taco",
            "description": description,
            "problem": problem,
            "tests": _translate_test_cases(ex["input_output"], time_limit=ex["time_limit"]),
        }

    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)
    return concatenate_datasets(processed_shards)
