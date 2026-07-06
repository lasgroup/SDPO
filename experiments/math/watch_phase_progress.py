#!/usr/bin/env python3
"""Watch one SDPO-Math experiment and print compact progress lines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


PROGRESS_KEYS = {
    "training/global_step": "step",
    "critic/score/mean": "score",
    "critic/rewards/mean": "reward",
    "val-core/math_dapo/acc/mean@1": "val_acc",
    "val-aux/math_dapo/incorrect_format/mean@1": "bad_fmt",
    "val-aux/math_dapo/truncated/mean@1": "trunc",
    "self_distillation/reprompt_sample_fraction": "reprompt",
    "self_distillation/feedback_used_fraction": "feedback",
    "self_distillation/reliability_weight_mean": "rel_w",
    "self_distillation/reliability_gate_threshold": "gate_thr",
    "self_distillation/reliability_gate_max_fraction": "gate_cap",
    "self_distillation/reliability_gate_eligible_fraction": "gate_eligible",
    "self_distillation/reliability_gate_target_fraction": "gate",
    "self_distillation/reliability_gate_compute_fraction": "gate_compute",
    "self_distillation/reliability_gate_compute_teacher_token_fraction": "gate_tok",
    "actor/pg_loss": "pg_loss",
    "actor/grad_norm": "grad",
    "perf/time_per_step": "step_s",
    "perf/throughput": "tok_s",
    "response_length/mean": "resp_tok",
    "response_length/clip_ratio": "resp_clip",
    "timing_s/gen": "gen_s",
    "timing_s/old_log_prob": "oldlp_s",
    "timing_s/update_actor": "upd_s",
    "timing_s/ema_teacher_update": "ema_s",
    "timing_s/adv": "adv_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--interval", default=10.0, type=float)
    parser.add_argument("--idle-interval", default=120.0, type=float)
    return parser.parse_args()


def metric_path(log_dir: Path, experiment_name: str) -> Path:
    return log_dir / "metrics" / "SDPO-Math" / f"{experiment_name}.jsonl"


def progress_path(log_dir: Path, experiment_name: str) -> Path:
    return log_dir / "metrics" / "SDPO-Math" / f"{experiment_name}.progress.jsonl"


def compact_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return str(value)


def progress_line(experiment_name: str, row: dict[str, Any], total_steps: int) -> str:
    data = row.get("data", {})
    step = data.get("training/global_step", row.get("step", "?"))
    if total_steps > 0:
        prefix = f"[progress] {experiment_name} step={step}/{total_steps}"
    else:
        prefix = f"[progress] {experiment_name} step={step}"

    parts = [prefix]
    for key, label in PROGRESS_KEYS.items():
        if key == "training/global_step":
            continue
        if key in data:
            parts.append(f"{label}={compact_number(data[key])}")
    return " ".join(parts)


def heartbeat_line(experiment_name: str, row: dict[str, Any], total_steps: int) -> str:
    step = row.get("step", "?")
    total = row.get("total_steps") or total_steps
    if total and total > 0:
        prefix = f"[progress] {experiment_name} step={step}/{total}"
    else:
        prefix = f"[progress] {experiment_name} step={step}"

    parts = [prefix, f"stage={row.get('event', 'unknown')}"]
    for key in ("validation", "model", "train_rows", "val_rows"):
        if key in row:
            parts.append(f"{key}={row[key]}")
    return " ".join(parts)


def read_jsonl_from(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset

    rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()

    for raw in lines:
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return rows, offset


def main() -> None:
    args = parse_args()
    metrics_file = metric_path(args.log_dir, args.experiment_name)
    progress_file = progress_path(args.log_dir, args.experiment_name)
    print(
        f"[progress] {args.experiment_name} waiting_for_progress={progress_file} "
        f"waiting_for_metrics={metrics_file}",
        flush=True,
    )

    metrics_offset = 0
    progress_offset = 0
    last_printed_step: Any = None
    last_printed_heartbeat: tuple[Any, Any, Any] | None = None
    last_activity = time.monotonic()

    while True:
        progress_rows, progress_offset = read_jsonl_from(progress_file, progress_offset)
        if progress_rows:
            latest = progress_rows[-1]
            heartbeat_key = (latest.get("step"), latest.get("event"), latest.get("validation"))
            if heartbeat_key != last_printed_heartbeat:
                print(heartbeat_line(args.experiment_name, latest, args.total_steps), flush=True)
                last_printed_heartbeat = heartbeat_key
                last_activity = time.monotonic()

        if not metrics_file.exists():
            now = time.monotonic()
            if now - last_activity >= args.idle_interval:
                print(f"[progress] {args.experiment_name} still_waiting_for_progress_or_metrics", flush=True)
                last_activity = now
            time.sleep(args.interval)
            continue

        rows, metrics_offset = read_jsonl_from(metrics_file, metrics_offset)

        printed = False
        for row in rows:
            data = row.get("data", {})
            step = data.get("training/global_step", row.get("step"))
            if step == last_printed_step and "training/global_step" in data:
                continue
            print(progress_line(args.experiment_name, row, args.total_steps), flush=True)
            last_printed_step = step
            last_activity = time.monotonic()
            printed = True

        if not printed:
            now = time.monotonic()
            if now - last_activity >= args.idle_interval:
                print(f"[progress] {args.experiment_name} no_new_metrics", flush=True)
                last_activity = now

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
