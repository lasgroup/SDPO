from datasets import concatenate_datasets, load_dataset, Dataset
import base64
import json
import pickle
import zlib
from datetime import datetime

from data.utils.code import parse_description

LCB_TEST_CUTOFF = datetime(2025, 2, 1)
LCB_TRAIN_CUTOFF = datetime(2025, 2, 1)
TIME_LIMIT = 6


def _parse_signature(starter_code: str) -> str:
    return "def " + starter_code.split("def ")[1].split("Input\n")[0].strip()


def _translate_private_test_cases(encoded_data, fn_name: str):
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    tests = json.loads(original_data)
    return json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "testtype": tests[0]["testtype"],
        "fn_name": fn_name,
        "time_limit": TIME_LIMIT,
    }, ensure_ascii=False)


def load_livecodebench(dataset_split: str, until: datetime = None) -> Dataset:
    ds = load_dataset("livecodebench/code_generation_lite", split="test", revision="refs/pr/6")  # see https://huggingface.co/datasets/livecodebench/code_generation_lite/discussions/5

    if dataset_split == "train":
        ds = ds.filter(lambda ex: ex["contest_date"] < LCB_TRAIN_CUTOFF)
    else:
        ds = ds.filter(lambda ex: ex["contest_date"] >= LCB_TEST_CUTOFF)
    if until is not None:
        ds = ds.filter(lambda ex: ex["contest_date"] < until)

    def format_prompt(ex):
        problem = ex["question_content"]  # already includes public test cases
        if ex["starter_code"].strip() != "":
            problem += f"\n\nYour solution should have the following signature: ```python\n{_parse_signature(ex['starter_code'])}\n```"

        if ex["metadata"].strip() != "":
            metadata = json.loads(ex["metadata"])
            fn_name = metadata.get("func_name", "")
        else:
            fn_name = ""

        description = parse_description(problem)

        return {
            "kind": "code",
            "dataset": "livecodebench",
            "description": description,
            "problem": problem,
            "tests": _translate_private_test_cases(ex["private_test_cases"], fn_name=fn_name),
        }

    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(format_prompt, remove_columns=ds.column_names, num_proc=4)
        processed_shards.append(shard)
    return concatenate_datasets(processed_shards)
