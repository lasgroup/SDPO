import datasets
import argparse
import pathlib


def split_tasks(json_path: str, output_dir: str, test_ratio: float = 0.1, seed: int = 42):
    ds = datasets.load_dataset("json", data_files=json_path, split="train")
    split_ds = ds.train_test_split(test_size=test_ratio, seed=seed)
    ds_train = split_ds["train"]
    ds_test = split_ds["test"]

    output_train_path = pathlib.Path(output_dir) / "train.json"
    output_test_path = pathlib.Path(output_dir) / "test.json"

    output_train_path.parent.mkdir(parents=True, exist_ok=True)
    output_test_path.parent.mkdir(parents=True, exist_ok=True)

    ds_train.to_json(output_train_path)
    ds_test.to_json(output_test_path)

    print(f"Saved split training set to: {output_train_path}")
    print(f"Saved split test set to: {output_test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the training set JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        required=True,
        help="Ratio of test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed for random number generator"
    )
    args = parser.parse_args()
    split_tasks(args.json_path, args.output_dir, args.test_ratio, args.seed)
