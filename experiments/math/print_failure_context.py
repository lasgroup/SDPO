#!/usr/bin/env python3
"""Print compact failure context for SDPO-Math quiet runs."""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path


ERROR_TOKENS = (
    "ERROR",
    "Error",
    "Exception",
    "Traceback",
    "RuntimeError",
    "ValueError",
    "ImportError",
    "ModuleNotFoundError",
    "CUDA",
    "out of memory",
    "OOM",
    "Killed",
    "EngineCore",
    "vLLM",
    "free memory",
    "No available memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-log", required=True, type=Path)
    parser.add_argument("--ray-log-root", default="/tmp/ray/session_latest/logs", type=Path)
    parser.add_argument("--since-epoch", type=float, default=0.0)
    parser.add_argument("--tail-lines", type=int, default=100)
    parser.add_argument("--max-ray-files", type=int, default=8)
    parser.add_argument("--max-blocks-per-file", type=int, default=3)
    return parser.parse_args()


def tail_lines(path: Path, limit: int) -> list[str]:
    rows: deque[str] = deque(maxlen=limit)
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                rows.append(line.rstrip())
    except OSError:
        return []
    return list(rows)


def has_error_token(line: str) -> bool:
    return any(token in line for token in ERROR_TOKENS)


def print_variant_tail(path: Path, tail: int) -> None:
    print("\n== failure context: variant log tail ==", flush=True)
    if not path.exists():
        print(f"missing variant log: {path}", flush=True)
        return
    print(f"path={path}", flush=True)
    for line in tail_lines(path, tail):
        print(line, flush=True)


def recent_ray_files(root: Path, since_epoch: float, max_files: int) -> list[Path]:
    if not root.exists():
        return []
    candidates: list[tuple[float, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if since_epoch and mtime < since_epoch - 30:
            continue
        name = path.name.lower()
        if not any(token in name for token in ("worker", "raylet", "python", "runtime_env", "dashboard")):
            continue
        candidates.append((mtime, path))
    candidates.sort(reverse=True)
    return [path for _, path in candidates[:max_files]]


def print_error_blocks(path: Path, max_blocks: int) -> bool:
    lines = tail_lines(path, 500)
    match_indices = [idx for idx, line in enumerate(lines) if has_error_token(line)]
    if not match_indices:
        return False

    print(f"\n## {path}", flush=True)
    printed_ranges: list[range] = []
    for idx in match_indices[-max_blocks:]:
        start = max(0, idx - 2)
        end = min(len(lines), idx + 16)
        current = range(start, end)
        if any(start in existing or end - 1 in existing for existing in printed_ranges):
            continue
        printed_ranges.append(current)
        for line in lines[start:end]:
            print(line, flush=True)
    return True


def print_ray_context(root: Path, since_epoch: float, max_files: int, max_blocks: int) -> None:
    print("\n== failure context: recent Ray/vLLM errors ==", flush=True)
    files = recent_ray_files(root, since_epoch, max_files)
    if not files:
        print(f"no recent Ray log files found under {root}", flush=True)
        return

    printed = False
    for path in files:
        printed = print_error_blocks(path, max_blocks) or printed
    if not printed:
        print(f"no error-looking lines found in recent Ray logs under {root}", flush=True)


def main() -> None:
    args = parse_args()
    print_variant_tail(args.variant_log, args.tail_lines)
    print_ray_context(args.ray_log_root, args.since_epoch, args.max_ray_files, args.max_blocks_per_file)


if __name__ == "__main__":
    main()
