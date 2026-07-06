#!/usr/bin/env python3
"""Write a structured manifest for an SDPO-Math phase run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--train-steps", required=True, type=int)
    parser.add_argument("--train-max-samples", required=True)
    parser.add_argument("--val-max-samples", required=True)
    parser.add_argument("--eval-freq", required=True)
    parser.add_argument("--save-freq", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exp-suffix", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    return parser.parse_args()


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    train_bs = os.environ.get("TRAIN_BS")
    rollout_n = os.environ.get("ROLLOUT_N")
    effective_rollouts = None
    if train_bs is not None and rollout_n is not None:
        try:
            effective_rollouts = int(train_bs) * int(rollout_n)
        except ValueError:
            effective_rollouts = None

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "project_root": os.environ.get("PROJECT_ROOT"),
        "phase": args.phase,
        "hardware_profile": os.environ.get("HARDWARE_PROFILE", "a100"),
        "profile": args.profile,
        "effective_rollouts_per_step": effective_rollouts,
        "profile_settings": {
            key.lower(): os.environ.get(key)
            for key in [
                "TRAIN_BS",
                "ROLLOUT_N",
                "ROLLOUT_TP",
                "AGENT_WORKERS",
                "RESPONSE_LEN",
                "MODEL_LEN",
                "ACTOR_LEN",
                "REPROMPT_LEN",
                "BATCHED_TOKENS",
                "MAX_NUM_SEQS",
                "GPU_UTIL",
                "SDPO_BATCHED_TOKENS",
                "SDPO_MAX_NUM_SEQS",
                "SDPO_GPU_UTIL",
                "SDPO_ACTOR_LEN",
                "SDPO_REPROMPT_LEN",
                "SDPO_ACTIVATION_OFFLOAD",
                "SDPO_DISTILLATION_TOPK",
                "RELIABILITY_GATE_MAX_FRACTION",
                "ENFORCE_EAGER",
                "ROLLOUT_QUANTIZATION",
            ]
        },
        "config_name": args.config_name,
        "model": args.model,
        "variants": args.variants.split(),
        "variant_hyperparameters": {
            "sdpo_vanilla": {
                "sparse_target_execution": os.environ.get("SDPO_SPARSE_TARGET_EXECUTION", "True").lower()
                == "true",
                "reliability_weighting": False,
            },
            "sdpo_reliability": {
                "reliability_gate_threshold": "0.0",
                "reliability_weighting": True,
                "sparse_target_execution": os.environ.get("SDPO_SPARSE_TARGET_EXECUTION", "True").lower()
                == "true",
            },
            "sdpo_reliability_gate": {
                "reliability_gate_threshold": os.environ.get("RELIABILITY_GATE_THRESHOLD", "0.4"),
                "reliability_gate_max_fraction": os.environ.get("RELIABILITY_GATE_MAX_FRACTION", "0.5"),
                "reliability_weighting": True,
                "reliability_gate_sparse_execution": os.environ.get(
                    "RELIABILITY_GATE_SPARSE_EXECUTION", "True"
                ).lower()
                == "true",
                "sparse_target_execution": os.environ.get("SDPO_SPARSE_TARGET_EXECUTION", "True").lower()
                == "true",
            }
        },
        "train_steps": args.train_steps,
        "train_max_samples": args.train_max_samples,
        "val_max_samples": args.val_max_samples,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "exp_suffix": args.exp_suffix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "log_dir": str(args.log_dir),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
