from datasets import Dataset
from datetime import datetime

from data.format.prompts import CODE_PROMPT
from data.format.utils import cast_large_strings
from data.utils.livecodebench import load_livecodebench
from data.utils.humaneval import load_humanevalplus
from data.utils.mbpp import load_mbppplus
from data.utils.codeforces import load_codeforces
from data.utils.code_elo import load_code_elo


def load_code(dataset_name: str) -> Dataset:
    assert dataset_name in ["open-r1/codeforces", "Qwen/CodeElo", "livecodebench/code_generation_lite-v6", "evalplus/humanevalplus", "evalplus/mbppplus"]

    if dataset_name == "open-r1/codeforces":
        ds = load_codeforces(dataset_split="test")
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
        ds = ds.map(lambda ex: {"prompt": CODE_PROMPT.format(problem=ex["problem"])})
    elif dataset_name == "Qwen/CodeElo":
        ds = load_code_elo()
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
    elif dataset_name == "livecodebench/code_generation_lite-v6":
        ds = load_livecodebench(dataset_split="test", until=datetime(2025, 5, 1))
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
        ds = ds.map(lambda ex: {"prompt": CODE_PROMPT.format(problem=ex["problem"])})
    elif dataset_name == "evalplus/humanevalplus":
        ds = load_humanevalplus()
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))
    elif dataset_name == "evalplus/mbppplus":
        ds = load_mbppplus()
        ds = cast_large_strings(ds, columns=list(ds.features.keys()))

    def check_description(ex):
        assert ex["description"].strip() != ""
    ds.map(check_description)

    return ds
