from datasets import Dataset, load_dataset
from typing import Optional, Dict
import argparse
from pathlib import Path


SYSTEM_PROMPT = """
Given a question and four options, please select the right answer. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, only output the letter corresponding to the correct option (A, B, C, or D), and nothing else. Do not restate the answer text. For example, if the answer is "A", just output:
<answer>
A
</answer>
"""


def format_choices(choices: Dict[str, str]) -> str:
    texts, labels = choices['text'], choices['label']
    return "\n".join([f"{label}: {text}" for label, text in zip(labels, texts)])

def format(row: dict) -> dict:
    return {
        "dataset": "sciknoweval",
        "system": SYSTEM_PROMPT,
        "prompt": row['question'] + "\n\n" + format_choices(row['choices']) + "\nPlease reason step by step.",
        "answer": row['answerKey'],

        # dummy fields
        "tests": None,
        "description": row['question'],
        "kind": "mcq",
        "elo": 1500,
    }

def load_sciknoweval(
    domain: Optional[str] = None,
    level: Optional[str] = None,
) -> Dataset:
    ds = load_dataset("hicai-zju/SciKnowEval", split='test')

    if domain:
        ds = ds.filter(lambda x: x['domain'] == domain)
    if level:
        ds = ds.filter(lambda x: x['details']['level'] == level)

    original_columns = ds.column_names
    return ds.map(format, remove_columns=original_columns)

