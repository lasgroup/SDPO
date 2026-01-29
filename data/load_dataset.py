import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset

from data.format.train import load_train
from data.format.gpqa import load_gpqa
from data.format.mmlu_pro import load_mmlu_pro
from data.format.math import load_math
from data.format.code import load_code
from data.format.sciknoweval import load_sciknoweval
from data.utils.data_handling import write_hf_to_json


def _add_embeddings(ds, embeddings_file = None):
    if not embeddings_file is None:
        embeddings = np.load(embeddings_file)
        ds = ds.add_column("embedding", [embeddings[i] for i in range(len(ds))])
    else:
        ds = ds.add_column("embedding", [np.array([])] * len(ds))
    return ds


def load_dataset_hf(
    dataset_name: str,
    output_path: str | None,
    start_idx: int = 0,
    num_el: int = None,
    category: str | None = None,
    embeddings_file: str | None = None,
) -> Dataset:

    final_columns = ["idx", "kind", "dataset", "answer", "elo", "prompt", "description", "tests", "embedding"]

    if category == "false":
        category = None

    if dataset_name == "lasgroup/verifiable-corpus":
        ds = load_train(category)
    elif dataset_name == "Idavidrein/gpqa-D":
        ds = load_gpqa(category)
    elif dataset_name == "TIGER-Lab/MMLU-Pro":
        ds = load_mmlu_pro(category, implementation="evalchemy")
    elif dataset_name in ["math-ai/aime24", "math-ai/aime25", "math-ai/math500", "math-ai/amc23", "openai/gsm8k"]:
        ds = load_math(dataset_name)
    elif dataset_name in ["open-r1/codeforces", "Qwen/CodeElo", "livecodebench/code_generation_lite-v6", "evalplus/humanevalplus", "evalplus/mbppplus"]:
        ds = load_code(dataset_name)
    elif dataset_name == "tooluse":
        print("Tooluse dataset is already loaded. You can proceed to preprocess it.")
    elif dataset_name in ["Biology", "Chemistry", "Material", "Physics"]:
        ds = load_sciknoweval(domain=dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    ds = ds.add_column("idx", list(range(len(ds))))
    ds = _add_embeddings(ds, embeddings_file=embeddings_file)


    # Common shape across datasets
    print(len(ds))
    if not "tests" in ds.column_names:
        ds = ds.add_column("tests", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"tests": "-" if ex["tests"] is None else ex["tests"]},  writer_batch_size=64)

    if not "answer" in ds.column_names:
        ds = ds.add_column("answer", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"answer": "-" if ex["answer"] is None else ex["answer"]})

    if not "dataset" in ds.column_names:
        ds = ds.add_column("dataset", [dataset_name] * len(ds))
    else:
        ds = ds.map(lambda ex: {"dataset": "-" if ex["dataset"] is None else ex["dataset"]})

    if not "elo" in ds.column_names:
        ds = ds.add_column("elo", ["-"] * len(ds))
    else:
        ds = ds.map(lambda ex: {"elo": "-" if ex["elo"] is None else ex["elo"]})

    # Add correct suffix to each description
    ds = ds.map(lambda ex: {"description": ex["description"] + f" The solution will be evaluated in a {ex['kind']} environment."})

    ds = ds.select_columns(final_columns)


    # Save dataset
    if num_el is None:
        num_el = len(ds)
    last_el = min(start_idx + num_el, len(ds))

    ds_filtered = ds.select(range(start_idx, last_el))
    print(ds_filtered.column_names)
    print(f"Export to file {output_path}.")
    print(ds_filtered.features)
    print(f"Final number dataset samples: {len(ds_filtered)}")

    if output_path is None:
        return ds_filtered
    else:
        write_hf_to_json(
            ds=ds_filtered,
            output_path=output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce sorted dataset that can be used for training on most relevant questions."
    )
    parser.add_argument(
        "--dataset_name", type=str,
        help="HF dataset name."
    )
    parser.add_argument(
        "--output_path", type=str,
        help="File where the dataset will be saved."
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index used from the final dataset."
    )
    parser.add_argument(
        "--num_el", type=int, default=None,
        help="End index used from the final dataset."
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="File where the dataset will be saved."
    )
    parser.add_argument(
        "--embeddings_file", type=str, default=None,
        help="Where the embeddings for the dataset lie."
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.0,
        help="Test ratio for the dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the dataset."
    )
    args = parser.parse_args()
    load_dataset_hf(
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        start_idx=args.start_idx,
        num_el=args.num_el,
        category=args.category,
        embeddings_file=args.embeddings_file,
    )
