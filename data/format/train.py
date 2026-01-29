from datasets import load_dataset, Dataset

from data.format.prompts import PROMPT, CODE_PROMPT
from data.format.utils import cast_large_strings


def _add_prompt(ex):
    if ex["kind"] == "code":
        return CODE_PROMPT.format(problem=ex["problem"])
    else:
        return PROMPT.format(problem=ex["problem"])


def load_train(category: str = None) -> Dataset:
    ds = load_dataset("lasgroup/verifiable-corpus", split="train")
    ds = cast_large_strings(ds, columns=list(ds.features.keys()))
    if not category is None:
        ds = ds.filter(lambda ex: ex["kind"] == category)
    return ds.map(lambda ex: {"prompt": _add_prompt(ex)})
