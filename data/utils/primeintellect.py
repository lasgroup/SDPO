from datasets import concatenate_datasets, load_dataset, Dataset
import ast
import json

from data.utils.code import parse_description, float_with_default

TIME_LIMIT = 1


def _parse_testtype(test):
    assert "type" in test
    if test["type"] == "stdin_stdout":
        return "stdin"
    elif test["type"] == "function_call":
        return "functional"
    return "stdin"


def _translate_test_cases(data):
    try:
        tests = ast.literal_eval(data)
    except (ValueError, SyntaxError) as e:
        try:  # try json loads instread
            tests = json.loads(data)
        except (json.JSONDecodeError, SyntaxError, ValueError) as e:
            print(repr(data))
            print(f"Error in json.loads: {e}")
    assert tests["language"] == "python"
    if isinstance(tests["test_cases"], list):
        out_tests = {
            "inputs": [t["input"] for t in tests["test_cases"]],
            "outputs": [t["output"] for t in tests["test_cases"]],
            "testtype": _parse_testtype(tests["test_cases"][0]),
            "fn_name": tests["test_cases"][0].get("fn_name", ""),
            "time_limit": TIME_LIMIT,
        }
    else:
        print(tests)
        raise ValueError(f"Unexpected type for tests: {type(tests)}")
    return json.dumps(out_tests, ensure_ascii=False)


def load_primeintellect(skip_without_solution=True) -> Dataset:
    ds = load_dataset("PrimeIntellect/verifiable-coding-problems", split="train")

    def has_valid_answer(ex):
        solution = ex.get("gold_standard_solution", "")
        if solution is None:
            return False
        return solution.startswith("```python") and solution.endswith("```")

    def format_prompt(ex):
        problem = ex["prompt"]  # already includes public test cases
        description = parse_description(problem)
        return {
            "kind": "code",
            "dataset": "primeintellect",
            "description": description,
            "problem": problem,
            "tests": _translate_test_cases(ex["verification_info"]),
        }

    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        if skip_without_solution:
            shard = shard.filter(has_valid_answer)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)
    return concatenate_datasets(processed_shards)
