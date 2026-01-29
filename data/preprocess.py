import os
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import argparse


def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):  return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):  return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):    return pa.field(field.name, pa.large_list(_to_large(pa.field("item", t.value_type)).type),
                                              field.nullable, field.metadata)
    if pa.types.is_struct(t):  return pa.field(field.name,
        pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
        field.nullable, field.metadata)
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    """Cast to LargeString/LargeList and write many small row groups.

    This avoids 32-bit offset overflow in Arrow arrays by casting to
    LargeString/LargeList and writing smaller row groups.
    """
    tbl: pa.Table = ds.data.table
    tbl = tbl.cast(_large_schema(tbl.schema))  # avoid 32-bit offset overflow
    # DO NOT combine_chunks() here — we want smaller arrays per row group
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


def make_map_fn(split: str):
        def process_fn(example, idx):
            question = example.pop("prompt")
            system = example.get("system", None)
            solution = example.pop("answer")
            global_id = example.pop("idx")

            tests = example.pop("tests")
            description = example.pop("description")
            reward_style = example.pop("kind")

            if split == "train" and "achievement_prior" in example.keys():
                achievement_prior = example.pop("achievement_prior")
            else:
                achievement_prior = 0

            data_source = example.pop("dataset")
            elo = example.pop("elo")

            if reward_style == "code":
                solution = tests

            # remove 'self' from method signature if present
            if data_source == "livecodebench":
                question = question.replace('(self, ', '(')
                description = description.replace('(self, ', '(')

            extra_info = {
                "split": split,
                "index": f"{global_id}",
                "description": description,
                "problem": question,
                "elo": elo,
                "achievement_prior": achievement_prior
            }

            if system is not None:
                messages = [{
                    "role": "system",
                    "content": system,
                }]
            else:
                messages = []
            return {
                "data_source": data_source, # this decides the reward function to use
                "prompt": messages + [{
                    "role": "user",
                    "content": question,
                }],
                "ability": reward_style,
                "reward_model": {"style": reward_style, "ground_truth": solution},
                "extra_info": extra_info,
            }

        return process_fn


def _map_in_shards(dataset, split: str, num_shards: int, num_proc: int):
    processed_shards = []
    for i in range(num_shards):
        shard = dataset.shard(num_shards=num_shards, index=i)
        shard = shard.map(function=make_map_fn(split), with_indices=True, num_proc=num_proc)
        processed_shards.append(shard)
    return datasets.concatenate_datasets(processed_shards)


def run_proprocessing(data_source, num_proc=4):
    print(data_source)
    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'train.json'), split='train')
    try:
        test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'test.json'), split='train')
    except:
        test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_source, 'test.json'), split='test')

    print(f"Map Datasets {train_dataset.num_rows} train, {test_dataset.num_rows} test")
    num_shards = min(4, (len(train_dataset) // 1000) + 1)
    print(f"Using {num_shards} shards")
    num_proc = min(num_proc, num_shards)
    print(f"Using {num_proc} processes")
    train_ds = _map_in_shards(train_dataset, "train", num_shards=num_shards, num_proc=num_proc)
    print(train_ds)
    test_ds = _map_in_shards(test_dataset, "test", num_shards=num_shards, num_proc=num_proc)
    print(test_ds)

    out_train = os.path.join(data_source, "train.parquet")
    out_test  = os.path.join(data_source, "test.parquet")
    write_rowgrouped_large(train_ds, out_train)
    write_rowgrouped_large(test_ds, out_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce sorted dataset that can be used for training on most relevant questions."
    )
    parser.add_argument(
        "--data_source", type=str,
        help="HF dataset name."
    )
    args = parser.parse_args()
    data_source = args.data_source
    run_proprocessing(data_source=data_source)
