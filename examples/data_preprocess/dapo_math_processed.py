# Copyright 2026 Individual Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Preprocess open-r1/DAPO-Math-17k-Processed for SDPO math training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import statistics
import tempfile
import unicodedata
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_DATASET_NAME = "open-r1/DAPO-Math-17k-Processed"
DEFAULT_SUBSET = "en"
DEFAULT_PROMPT_SUFFIX = (
    "Give a concise solution with only the necessary reasoning; do not restate the problem or repeat "
    "calculations. End with exactly one final answer in \\boxed{}."
)
DEFAULT_DATA_SOURCE = "math_dapo"
PROMPT_STYLE = "concise_v1"
DEFAULT_EVAL_DATASETS = ("MathArena/aime_2025", "MathArena/paper_benchmark")
EVAL_PROBLEM_FIELDS = ("problem", "prompt", "question", "description")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert open-r1/DAPO-Math-17k-Processed to verl RL parquet format."
    )
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument(
        "--local_parquet_path",
        default=None,
        help="Optional local source parquet path. If set, avoids loading from Hugging Face.",
    )
    parser.add_argument("--local_save_dir", default="data/dapo_math_en")
    parser.add_argument(
        "--update_prepared_dir",
        default=None,
        help="Update prompt text in existing train/val parquet files, then exit.",
    )
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--validation_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--prompt_suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--feedback_mode", choices=["none", "safe", "oracle"], default="safe")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decontaminate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--eval_dataset_names",
        nargs="*",
        default=list(DEFAULT_EVAL_DATASETS),
        help="HF eval datasets for exact-overlap decontamination when `datasets` is installed.",
    )
    parser.add_argument(
        "--eval_parquet_paths",
        nargs="*",
        default=[],
        help="Local eval parquet files. These are used in addition to --eval_dataset_names.",
    )
    parser.add_argument("--ngram_size", type=int, default=5)
    parser.add_argument("--ngram_jaccard_threshold", type=float, default=0.70)
    parser.add_argument("--near_duplicate_top_k", type=int, default=100)
    parser.add_argument(
        "--remove_near_duplicates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove train rows whose n-gram Jaccard similarity to any eval prompt exceeds the threshold.",
    )
    return parser.parse_args()


def load_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.local_parquet_path:
        return pq.read_table(args.local_parquet_path).to_pylist()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading directly from Hugging Face requires the `datasets` package. "
            "Install it or pass --local_parquet_path."
        ) from exc

    return [dict(row) for row in load_dataset(args.dataset_name, args.subset, split="train")]


def normalize_problem_text(text: str) -> str:
    """Normalize problem text for deterministic exact-match dedup/decontamination."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")

    instruction_patterns = [
        r"solve the following math problem step by step\.",
        r"the last line of your response should be of the form answer:.*?(?=\n\n|$)",
        r"remember to put your answer on its own line after [\"']?answer:[\"']?\.?",
        r"please reason step by step,? and put your final answer within \\boxed\{\}\.?",
        r"let'?s think step by step and output the final answer within \\boxed\{\}\.?",
        re.escape(DEFAULT_PROMPT_SUFFIX.lower()),
    ]
    for pattern in instruction_patterns:
        text = re.sub(pattern, " ", text, flags=re.DOTALL)

    text = re.sub(r"\s+", " ", text).strip()
    # Keep mathematical punctuation but remove purely typographic noise.
    text = text.replace("`", "").replace('"', "").replace("'", "")
    return re.sub(r"\s+", "", text)


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for record in records:
        raw_prompt = record["extra_info"]["raw_prompt"]
        norm = normalize_problem_text(raw_prompt)
        index = record["extra_info"]["index"]
        if norm in seen:
            duplicate = dict(record)
            duplicate["extra_info"] = dict(record["extra_info"])
            duplicate["extra_info"]["duplicate_of"] = seen[norm]
            removed.append(duplicate)
            continue
        seen[norm] = index
        kept.append(record)
    return kept, removed


def get_eval_problem(row: dict[str, Any]) -> str:
    for field in EVAL_PROBLEM_FIELDS:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_eval_rows_from_parquet(path: str) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    eval_rows = []
    for idx, row in enumerate(rows):
        problem = get_eval_problem(row)
        if not problem:
            continue
        eval_rows.append(
            {
                "source": Path(path).stem,
                "index": str(row.get("problem_idx", idx)),
                "problem": problem,
            }
        )
    return eval_rows


def load_eval_rows_from_hf(dataset_name: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading eval datasets from Hugging Face requires the `datasets` package. "
            "Pass --eval_parquet_paths or install `datasets`."
        ) from exc

    rows = [dict(row) for row in load_dataset(dataset_name, split="train")]
    eval_rows = []
    for idx, row in enumerate(rows):
        problem = get_eval_problem(row)
        if not problem:
            continue
        eval_rows.append(
            {
                "source": dataset_name,
                "index": str(row.get("problem_idx", idx)),
                "problem": problem,
            }
        )
    return eval_rows


def load_eval_problem_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for path in args.eval_parquet_paths:
        eval_rows.extend(load_eval_rows_from_parquet(path))
    if eval_rows:
        return eval_rows

    for dataset_name in args.eval_dataset_names:
        eval_rows.extend(load_eval_rows_from_hf(dataset_name))
    return eval_rows


def remove_exact_eval_overlaps(
    records: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_by_norm: dict[str, list[dict[str, Any]]] = {}
    for row in eval_rows:
        norm = normalize_problem_text(row["problem"])
        eval_by_norm.setdefault(norm, []).append(row)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        norm = normalize_problem_text(record["extra_info"]["raw_prompt"])
        matches = eval_by_norm.get(norm)
        if not matches:
            kept.append(record)
            continue

        contaminated = dict(record)
        contaminated["extra_info"] = dict(record["extra_info"])
        contaminated["extra_info"]["contamination_matches"] = matches
        contaminated["extra_info"]["contamination_type"] = "exact_normalized"
        removed.append(contaminated)
    return kept, removed


def make_ngrams(text: str, ngram_size: int) -> set[str]:
    normalized = normalize_problem_text(text)
    if not normalized:
        return set()
    if len(normalized) <= ngram_size:
        return {normalized}
    return {normalized[i : i + ngram_size] for i in range(len(normalized) - ngram_size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def audit_near_eval_overlaps(
    records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    ngram_size: int,
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    eval_ngrams = [
        {
            "source": row["source"],
            "index": row["index"],
            "problem": row["problem"],
            "ngrams": make_ngrams(row["problem"], ngram_size),
        }
        for row in eval_rows
    ]

    candidates: list[dict[str, Any]] = []
    for record in records:
        train_ngrams = make_ngrams(record["extra_info"]["raw_prompt"], ngram_size)
        best_score = 0.0
        best_eval: dict[str, Any] | None = None
        for eval_row in eval_ngrams:
            score = jaccard(train_ngrams, eval_row["ngrams"])
            if score > best_score:
                best_score = score
                best_eval = eval_row

        if best_eval is not None and best_score >= threshold:
            candidates.append(
                {
                    "train_index": record["extra_info"]["index"],
                    "eval_source": best_eval["source"],
                    "eval_index": best_eval["index"],
                    "jaccard": round(best_score, 6),
                    "train_prompt": record["extra_info"]["raw_prompt"],
                    "eval_problem": best_eval["problem"],
                }
            )

    candidates.sort(key=lambda row: row["jaccard"], reverse=True)
    return candidates[:top_k]


def remove_near_eval_overlaps(
    records: list[dict[str, Any]], near_candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexes_to_remove = {candidate["train_index"] for candidate in near_candidates}
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for record in records:
        if record["extra_info"]["index"] not in indexes_to_remove:
            kept.append(record)
            continue

        near_duplicate = dict(record)
        near_duplicate["extra_info"] = dict(record["extra_info"])
        near_duplicate["extra_info"]["contamination_type"] = "ngram_jaccard"
        near_duplicate["extra_info"]["near_duplicate_candidates"] = [
            candidate for candidate in near_candidates if candidate["train_index"] == record["extra_info"]["index"]
        ]
        removed.append(near_duplicate)
    return kept, removed


def get_ground_truth(row: dict[str, Any]) -> str:
    solution = str(row.get("solution") or "").strip()
    reward_model = row.get("reward_model") or {}
    reward_ground_truth = str(reward_model.get("ground_truth") or "").strip()
    return solution or reward_ground_truth


def format_prompt(problem: str, prompt_suffix: str) -> str:
    problem = problem.strip()
    prompt_suffix = prompt_suffix.strip()
    return f"{problem}\n\n{prompt_suffix}"


def prepared_prompt_style_is_current(data_dir: Path, prompt_suffix: str) -> bool:
    metadata_path = data_dir / "prompt_style.json"
    parquet_missing = any(not (data_dir / f"{split}.parquet").exists() for split in ("train", "val"))
    if not metadata_path.exists() or parquet_missing:
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return metadata.get("style") == PROMPT_STYLE and metadata.get("prompt_suffix") == prompt_suffix


def update_prepared_prompts(data_dir: Path, prompt_suffix: str) -> dict[str, dict[str, int]]:
    if prepared_prompt_style_is_current(data_dir, prompt_suffix):
        return {
            split: {"rows": pq.ParquetFile(data_dir / f"{split}.parquet").metadata.num_rows, "changed": 0}
            for split in ("train", "val")
        }

    results: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)

        table = pq.read_table(path)
        rows = table.to_pylist()
        changed = 0
        for row in rows:
            raw_prompt = str((row.get("extra_info") or {}).get("raw_prompt") or "").strip()
            if not raw_prompt:
                raise ValueError(f"{path} contains a row without extra_info.raw_prompt")
            prompt = [{"role": "user", "content": format_prompt(raw_prompt, prompt_suffix)}]
            if row.get("prompt") != prompt:
                row["prompt"] = prompt
                changed += 1

        if changed:
            updated_table = pa.Table.from_pylist(rows, schema=table.schema)
            with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".parquet", delete=False) as handle:
                temporary_path = Path(handle.name)
            try:
                pq.write_table(updated_table, temporary_path, compression="snappy")
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
        results[split] = {"rows": len(rows), "changed": changed}

    metadata = {
        "style": PROMPT_STYLE,
        "prompt_suffix": prompt_suffix,
        "splits": results,
    }
    write_json(data_dir / "prompt_style.json", metadata)
    return results


def get_original_index(row: dict[str, Any], fallback_idx: int) -> str:
    extra_info = row.get("extra_info") or {}
    index = extra_info.get("index", fallback_idx)
    return str(index)


def convert_row(row: dict[str, Any], row_idx: int, args: argparse.Namespace) -> dict[str, Any] | None:
    raw_problem = str(row.get("prompt") or "").strip()
    ground_truth = get_ground_truth(row)
    if not raw_problem or not ground_truth:
        return None

    original_index = get_original_index(row, row_idx)
    return {
        "data_source": args.data_source,
        "prompt": [{"role": "user", "content": format_prompt(raw_problem, args.prompt_suffix)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": "train",
            "index": original_index,
            "raw_prompt": raw_problem,
            "feedback_mode": args.feedback_mode,
        },
    }


def convert_rows(source_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter]:
    drop_reasons: Counter = Counter()
    converted: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        record = convert_row(row, idx, args)
        if record is None:
            if not str(row.get("prompt") or "").strip():
                drop_reasons["empty_prompt"] += 1
            if not get_ground_truth(row):
                drop_reasons["empty_ground_truth"] += 1
            continue
        converted.append(record)
    return converted, drop_reasons


def split_train_val(
    records: list[dict[str, Any]], validation_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if validation_size < 0:
        raise ValueError("--validation_size must be non-negative")
    if validation_size >= len(records):
        raise ValueError("--validation_size must be smaller than the number of converted records")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_records = shuffled[:validation_size]
    train_records = shuffled[validation_size:]

    for record in val_records:
        record["extra_info"]["split"] = "val"
    return train_records, val_records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), str(path), compression="zstd")


def quantiles(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": ordered[int(0.90 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def answer_kind(answer: str) -> str:
    if re.fullmatch(r"-?\d+", answer):
        return "integer"
    if re.fullmatch(r"-?\d+\.\d+", answer):
        return "decimal"
    if "\\frac" in answer or "/" in answer:
        return "fraction"
    if "\\sqrt" in answer or "sqrt" in answer:
        return "radical"
    if "," in answer or ";" in answer:
        return "tuple_or_list"
    if re.search(r"[A-Za-z]", answer):
        return "symbolic_or_text"
    return "other"


def package_status(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "missing"


def build_report(
    args: argparse.Namespace,
    source_rows: list[dict[str, Any]],
    converted_rows: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    drop_reasons: Counter,
    duplicate_records: list[dict[str, Any]],
    exact_contaminated_records: list[dict[str, Any]],
    near_contaminated_records: list[dict[str, Any]],
    near_duplicate_candidates: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> str:
    raw_prompt_lengths = [len(str(row.get("prompt") or "")) for row in source_rows]
    formatted_prompt_lengths = [len(row["prompt"][0]["content"]) for row in converted_rows]
    answers = [row["reward_model"]["ground_truth"] for row in converted_rows]
    answer_lengths = [len(answer) for answer in answers]
    answer_kinds = Counter(answer_kind(answer) for answer in answers)
    answer_prompt_count = sum("Answer:" in row["prompt"][0]["content"] for row in converted_rows)

    sample_record = train_records[0] if train_records else converted_rows[0]
    sample_prompt = sample_record["prompt"][0]["content"]

    lines = [
        "# DAPO Math Data Report",
        "",
        "## Stage 0 Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- `datasets`: {package_status('datasets')}",
        f"- `pyarrow`: {package_status('pyarrow')}",
        f"- `math-verify`: {package_status('math-verify')}",
        "- Model runtime/GPU loading: not validated by this data preprocessing step.",
        "",
        "## Source",
        "",
        f"- Dataset: `{args.dataset_name}`",
        f"- Subset: `{args.subset}`",
        f"- Local parquet: `{args.local_parquet_path or ''}`",
        f"- Output data source: `{args.data_source}`",
        f"- Feedback mode: `{args.feedback_mode}`",
        f"- Seed: `{args.seed}`",
        f"- Validation size: `{args.validation_size}`",
        f"- Max samples: `{args.max_samples}`",
        "",
        "## Counts",
        "",
        f"- Raw rows loaded: {len(source_rows)}",
        f"- Converted rows: {len(converted_rows)}",
        f"- Removed duplicates: {len(duplicate_records)}",
        f"- Eval rows loaded for decontamination: {len(eval_rows)}",
        f"- Removed exact eval overlaps: {len(exact_contaminated_records)}",
        f"- Near-duplicate candidates above threshold: {len(near_duplicate_candidates)}",
        f"- Removed near-duplicate overlaps: {len(near_contaminated_records)}",
        f"- Train rows: {len(train_records)}",
        f"- Validation rows: {len(val_records)}",
        f"- Dropped rows: {sum(drop_reasons.values())}",
        f"- Drop reasons: `{dict(drop_reasons)}`",
        f"- Prompts containing `Answer:` after conversion: {answer_prompt_count}",
        "",
        "## Length Statistics",
        "",
        f"- Raw prompt chars: `{quantiles(raw_prompt_lengths)}`",
        f"- Formatted prompt chars: `{quantiles(formatted_prompt_lengths)}`",
        f"- Answer chars: `{quantiles(answer_lengths)}`",
        "",
        "## Answer Kinds",
        "",
        "```text",
        json.dumps(dict(answer_kinds), indent=2, sort_keys=True),
        "```",
        "",
        "## Sample Converted Prompt",
        "",
        "```text",
        sample_prompt[:2000],
        "```",
        "",
        "## Sample Ground Truth",
        "",
        "```text",
        sample_record["reward_model"]["ground_truth"],
        "```",
        "",
    ]
    return "\n".join(lines)


def build_decontamination_report(
    args: argparse.Namespace,
    raw_count: int,
    converted_count: int,
    deduped_count: int,
    final_pool_count: int,
    duplicate_records: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    exact_contaminated_records: list[dict[str, Any]],
    near_duplicate_candidates: list[dict[str, Any]],
    near_contaminated_records: list[dict[str, Any]],
) -> str:
    eval_counts = Counter(row["source"] for row in eval_rows)
    lines = [
        "# Decontamination Report",
        "",
        "## Configuration",
        "",
        f"- Deduplicate: `{args.deduplicate}`",
        f"- Decontaminate: `{args.decontaminate}`",
        f"- Eval dataset names: `{args.eval_dataset_names}`",
        f"- Eval parquet paths: `{args.eval_parquet_paths}`",
        f"- N-gram size: `{args.ngram_size}`",
        f"- N-gram Jaccard threshold: `{args.ngram_jaccard_threshold}`",
        f"- Remove near duplicates: `{args.remove_near_duplicates}`",
        "",
        "## Counts",
        "",
        f"- Raw train samples: {raw_count}",
        f"- Converted train samples: {converted_count}",
        f"- Removed by deduplication: {len(duplicate_records)}",
        f"- After deduplication: {deduped_count}",
        f"- Eval samples loaded: {len(eval_rows)}",
        f"- Eval samples by source: `{dict(eval_counts)}`",
        f"- Removed by exact normalized overlap: {len(exact_contaminated_records)}",
        f"- N-gram near-duplicate candidates above threshold: {len(near_duplicate_candidates)}",
        f"- Removed by n-gram near-duplicate rule: {len(near_contaminated_records)}",
        f"- Final clean pool before validation split: {final_pool_count}",
        "",
        "## Top N-Gram Candidates",
        "",
    ]

    if not near_duplicate_candidates:
        lines.append("No candidates above threshold.")
    else:
        lines.extend(["| Train Index | Eval Source | Eval Index | Jaccard |", "|---|---|---|---:|"])
        for candidate in near_duplicate_candidates[:20]:
            lines.append(
                f"| `{candidate['train_index']}` | `{candidate['eval_source']}` | "
                f"`{candidate['eval_index']}` | {candidate['jaccard']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.update_prepared_dir:
        results = update_prepared_prompts(Path(args.update_prepared_dir), args.prompt_suffix)
        print("prepared_prompt_ok:", {"style": PROMPT_STYLE, "splits": results})
        return

    save_dir = Path(args.local_save_dir)
    report_dir = Path(args.report_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_source_rows(args)
    if args.max_samples > 0:
        source_rows = source_rows[: args.max_samples]

    converted_rows, drop_reasons = convert_rows(source_rows, args)
    if args.deduplicate:
        deduped_rows, duplicate_records = deduplicate_records(converted_rows)
    else:
        deduped_rows, duplicate_records = converted_rows, []

    eval_rows: list[dict[str, Any]] = []
    exact_contaminated_records: list[dict[str, Any]] = []
    near_duplicate_candidates: list[dict[str, Any]] = []
    near_contaminated_records: list[dict[str, Any]] = []
    clean_pool = deduped_rows
    if args.decontaminate:
        eval_rows = load_eval_problem_rows(args)
        clean_pool, exact_contaminated_records = remove_exact_eval_overlaps(clean_pool, eval_rows)
        near_duplicate_candidates = audit_near_eval_overlaps(
            records=clean_pool,
            eval_rows=eval_rows,
            ngram_size=args.ngram_size,
            threshold=args.ngram_jaccard_threshold,
            top_k=args.near_duplicate_top_k,
        )
        if args.remove_near_duplicates:
            clean_pool, near_contaminated_records = remove_near_eval_overlaps(clean_pool, near_duplicate_candidates)

    train_records, val_records = split_train_val(clean_pool, args.validation_size, args.seed)

    write_jsonl(save_dir / "train_raw.jsonl", source_rows)
    write_jsonl(save_dir / "train_dedup.jsonl", deduped_rows)
    write_jsonl(save_dir / "removed_duplicates.jsonl", duplicate_records)
    write_jsonl(save_dir / "removed_contaminated.jsonl", exact_contaminated_records + near_contaminated_records)
    write_jsonl(save_dir / "near_duplicate_candidates.jsonl", near_duplicate_candidates)
    write_jsonl(save_dir / "train_clean.jsonl", train_records)
    write_jsonl(save_dir / "val_clean.jsonl", val_records)
    write_json(save_dir / "train_example.json", train_records[0])
    write_json(save_dir / "val_example.json", val_records[0])
    write_parquet(save_dir / "train.parquet", train_records)
    write_parquet(save_dir / "val.parquet", val_records)

    report = build_report(
        args=args,
        source_rows=source_rows,
        converted_rows=converted_rows,
        train_records=train_records,
        val_records=val_records,
        drop_reasons=drop_reasons,
        duplicate_records=duplicate_records,
        exact_contaminated_records=exact_contaminated_records,
        near_contaminated_records=near_contaminated_records,
        near_duplicate_candidates=near_duplicate_candidates,
        eval_rows=eval_rows,
    )
    (report_dir / "dapo_math_data_report.md").write_text(report, encoding="utf-8")
    decontamination_report = build_decontamination_report(
        args=args,
        raw_count=len(source_rows),
        converted_count=len(converted_rows),
        deduped_count=len(deduped_rows),
        final_pool_count=len(clean_pool),
        duplicate_records=duplicate_records,
        eval_rows=eval_rows,
        exact_contaminated_records=exact_contaminated_records,
        near_duplicate_candidates=near_duplicate_candidates,
        near_contaminated_records=near_contaminated_records,
    )
    (report_dir / "decontamination_report.md").write_text(decontamination_report, encoding="utf-8")

    print(f"Loaded rows: {len(source_rows)}")
    print(f"Converted rows: {len(converted_rows)}")
    print(f"Deduplicated rows: {len(deduped_rows)}")
    print(f"Removed duplicates: {len(duplicate_records)}")
    print(f"Eval rows: {len(eval_rows)}")
    print(f"Removed exact eval overlaps: {len(exact_contaminated_records)}")
    print(f"Near-duplicate candidates: {len(near_duplicate_candidates)}")
    print(f"Removed near-duplicate overlaps: {len(near_contaminated_records)}")
    print(f"Train rows: {len(train_records)}")
    print(f"Validation rows: {len(val_records)}")
    print(f"Wrote data to: {save_dir}")
    print(f"Wrote report to: {report_dir / 'dapo_math_data_report.md'}")


if __name__ == "__main__":
    main()
