#!/usr/bin/env python3
"""Print a compact SDPO-Math phase log summary."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


KEY_TOKENS = [
    "training/global_step",
    "val-core",
    "val-aux",
    "reward",
    "score",
    "acc",
    "format",
    "truncated",
    "self_distillation",
    "actor/grad_norm",
    "actor/pg_loss",
    "perf/throughput",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--tail", default=3, type=int)
    return parser.parse_args()


def print_console_log_tail(log_dir: Path, tail: int) -> None:
    for path in sorted(log_dir.glob("*.log")):
        print(f"\n## {path.name}")
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if "step:" in line
        ]
        for line in lines[-tail:]:
            parts = line.replace(" - ", "\n").splitlines()
            for part in parts:
                if any(token in part for token in KEY_TOKENS) or part.startswith("step:"):
                    print(f"  {part}")


def print_file_logger_tail(log_dir: Path, tail: int) -> None:
    metric_files = sorted(glob.glob(str(log_dir / "metrics" / "SDPO-Math" / "*.jsonl")))
    if not metric_files:
        return

    print("\nfile_logger_metrics:")
    for file_name in metric_files:
        path = Path(file_name)
        print(f"\n## {path.name}")
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        for row in rows[-tail:]:
            data = row.get("data", {})
            print(f"step:{row.get('step')}")
            for key, value in sorted(data.items()):
                if any(token in key for token in KEY_TOKENS):
                    print(f"  {key}:{value}")


def print_checkpoints() -> None:
    print("\ncheckpoints:")
    paths = sorted(path for path in Path("checkpoints/sdpo_math").rglob("global_step_*") if path.is_dir())
    for path in paths[-20:]:
        print(path)


def main() -> None:
    args = parse_args()
    if not args.log_dir.exists():
        raise SystemExit(f"missing log dir: {args.log_dir}")

    print(f"log_dir={args.log_dir}")
    print_console_log_tail(args.log_dir, args.tail)
    print_file_logger_tail(args.log_dir, args.tail)
    print_checkpoints()


if __name__ == "__main__":
    main()
