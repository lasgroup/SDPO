import os
import json
import numpy as np
import datasets
import copy
import argparse

PERCENTAGE_TO_KEEP = 0.5


def sample_tests(example):
    tests = json.loads(example["tests"])
    inputs = tests["inputs"]
    outputs = tests["outputs"]

    num_tests = len(inputs)
    keep_count = max(1, int(num_tests * PERCENTAGE_TO_KEEP))
    keep_indices = np.sort(
        np.random.choice(num_tests, size=keep_count, replace=False)
    )

    reduced_inputs = [inputs[i] for i in keep_indices]
    reduced_outputs = [outputs[i] for i in keep_indices]

    reduced_tests = copy.deepcopy(tests)
    reduced_tests["inputs"] = reduced_inputs
    reduced_tests["outputs"] = reduced_outputs

    example["tests"] = json.dumps(reduced_tests)
    return example


def main(data_path, output_dir):
    np.random.seed(0)

    ds_train = datasets.load_dataset("json", data_files=data_path, split="train")

    # Save full dataset as test.json
    test_file = os.path.join(output_dir, "test.json")
    ds_train.to_json(test_file)

    # Create reduced dataset with 50% of tests
    ds_train_reduced = ds_train.map(sample_tests)

    # Save reduced dataset as train.json
    train_file = os.path.join(output_dir, "train.json")
    ds_train_reduced.to_json(train_file)

    # check counts
    orig_counts = []
    reduced_counts = []
    for orig_item, new_item in zip(ds_train, ds_train_reduced):
        orig_tests = json.loads(orig_item["tests"]) if isinstance(orig_item["tests"], str) else orig_item["tests"]
        new_tests = json.loads(new_item["tests"]) if isinstance(new_item["tests"], str) else new_item["tests"]

        orig_counts.append(len(orig_tests["inputs"]))
        reduced_counts.append(len(new_tests["inputs"]))

    print(f"Saved full dataset (test set) to: {test_file}")
    print(f"Saved reduced dataset (train set) to: {train_file}")
    print(f"Share of tests kept in train set: {PERCENTAGE_TO_KEEP:.2f}")
    print(f"Mean tests per item: {np.mean(orig_counts):.2f} → {np.mean(reduced_counts):.2f}")


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
    args = parser.parse_args()
    main(args.json_path, args.output_dir)
