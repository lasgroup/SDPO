#!/usr/bin/env python3
"""Validate that an SDPO-Math phase run has enough structured outputs for reporting."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED_VARIANTS = {"base_rl", "sdpo_vanilla", "sdpo_reliability_gate"}
OPTIONAL_VARIANTS = {"sdpo_reliability"}
ALLOWED_VARIANTS = REQUIRED_VARIANTS | OPTIONAL_VARIANTS
TRAINED_VARIANTS = ALLOWED_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--expect-phase")
    parser.add_argument("--expect-model")
    parser.add_argument("--expect-profile")
    parser.add_argument("--expect-seed", type=int)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def latest_checkpoint_dir(root: Path) -> Path | None:
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        step = tracker.read_text(encoding="utf-8").strip()
        candidate = root / f"global_step_{step}"
        if candidate.exists():
            return candidate

    candidates = [path for path in root.glob("global_step_*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def main() -> None:
    args = parse_args()
    manifest_path = args.log_dir / "manifest.json"
    summary_path = args.log_dir / "summary.csv"
    require(manifest_path.exists(), f"missing {manifest_path}")
    require(summary_path.exists(), f"missing {summary_path}; run summarize_phase_results.py first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project_root = Path(manifest.get("project_root") or ".")
    manifest_variants = set(manifest["variants"])
    require(REQUIRED_VARIANTS <= manifest_variants, f"manifest missing required variants: {manifest['variants']}")
    require(manifest_variants <= ALLOWED_VARIANTS, f"unexpected variants in manifest: {manifest['variants']}")
    require(manifest["seed"] is not None, "manifest missing seed")
    require(manifest["model"], "manifest missing model")
    require(manifest.get("config_name") == "sdpo_math_a100", f"unexpected config_name: {manifest.get('config_name')}")
    require(manifest.get("profile_settings"), "manifest missing profile_settings")
    require(manifest.get("effective_rollouts_per_step"), "manifest missing effective_rollouts_per_step")
    for variant in {"sdpo_vanilla", "sdpo_reliability", "sdpo_reliability_gate"} & manifest_variants:
        variant_cfg = manifest.get("variant_hyperparameters", {}).get(variant, {})
        require(
            variant_cfg.get("sparse_target_execution") is True,
            f"manifest missing {variant} sparse_target_execution=True",
        )
    if "sdpo_reliability" in manifest_variants:
        reliability_cfg = manifest.get("variant_hyperparameters", {}).get("sdpo_reliability", {})
        require(
            reliability_cfg.get("reliability_weighting") is True,
            "manifest missing sdpo_reliability reliability_weighting=True",
        )
        require(
            str(reliability_cfg.get("reliability_gate_threshold")) == "0.0",
            "manifest missing sdpo_reliability reliability_gate_threshold=0.0",
        )
    gate_cfg = manifest.get("variant_hyperparameters", {}).get("sdpo_reliability_gate", {})
    require(
        gate_cfg.get("reliability_weighting") is True,
        "manifest missing sdpo_reliability_gate reliability_weighting=True",
    )
    require(
        gate_cfg.get("reliability_gate_threshold") not in (None, ""),
        "manifest missing sdpo_reliability_gate reliability_gate_threshold",
    )
    require(
        gate_cfg.get("reliability_gate_max_fraction") not in (None, ""),
        "manifest missing sdpo_reliability_gate reliability_gate_max_fraction",
    )
    require(
        gate_cfg.get("reliability_gate_sparse_execution") is True,
        "manifest missing sdpo_reliability_gate sparse execution",
    )
    if args.expect_phase:
        require(manifest.get("phase") == args.expect_phase, f"unexpected phase: {manifest.get('phase')}")
    if args.expect_model:
        require(manifest.get("model") == args.expect_model, f"unexpected model: {manifest.get('model')}")
    if args.expect_profile:
        require(manifest.get("profile") == args.expect_profile, f"unexpected profile: {manifest.get('profile')}")
    if args.expect_seed is not None:
        require(manifest.get("seed") == args.expect_seed, f"unexpected seed: {manifest.get('seed')}")

    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    variants = {row["variant"] for row in rows}
    require(variants == manifest_variants, f"summary variants mismatch: {variants}")

    for row in rows:
        variant = row["variant"]
        require(row["val_acc_mean"] != "", f"{variant} missing val_acc_mean")
        require(row["incorrect_format_mean"] != "", f"{variant} missing incorrect_format_mean")
        require(row["truncated_mean"] != "", f"{variant} missing truncated_mean")
        if variant.startswith("sdpo_"):
            require(row["sdpo_reprompt_fraction"] != "", f"{variant} missing SDPO reprompt metric")
            require(row["sdpo_feedback_used_fraction"] != "", f"{variant} missing SDPO feedback-used metric")
        if variant in {"sdpo_reliability", "sdpo_reliability_gate"}:
            require(row["sdpo_reliability_weight_mean"] != "", f"{variant} missing reliability weight metric")
        if variant == "sdpo_reliability_gate":
            require(
                row.get("sdpo_reliability_gate_threshold", "") != "",
                "sdpo_reliability_gate missing gate threshold metric",
            )
            require(
                row.get("sdpo_reliability_gate_max_fraction", "") != "",
                "sdpo_reliability_gate missing gate max-fraction metric",
            )
            require(
                row.get("sdpo_reliability_gate_eligible_fraction", "") != "",
                "sdpo_reliability_gate missing eligible-fraction metric",
            )
            require(
                row.get("sdpo_reliability_gate_fraction", "") != "",
                "sdpo_reliability_gate missing gate fraction metric",
            )
            require(
                row.get("sdpo_reliability_gate_compute_fraction", "") != "",
                "sdpo_reliability_gate missing gate compute fraction metric",
            )
            require(
                row.get("sdpo_reliability_gate_compute_token_fraction", "") != "",
                "sdpo_reliability_gate missing gate compute token fraction metric",
            )

        validation_dir = args.log_dir / "validation" / f"{variant}_{manifest['exp_suffix']}"
        require(validation_dir.exists(), f"{variant} missing validation dump dir: {validation_dir}")
        require(list(validation_dir.glob("*.jsonl")), f"{variant} missing validation jsonl dumps: {validation_dir}")

    if args.require_checkpoints:
        exp_suffix = manifest["exp_suffix"]
        expected_step = int(manifest["train_steps"])
        for variant in manifest_variants & TRAINED_VARIANTS:
            ckpt_root = project_root / "checkpoints/sdpo_math" / f"{variant}_{exp_suffix}"
            require(ckpt_root.exists(), f"missing checkpoint root for {variant}: {ckpt_root}")
            latest_ckpt = latest_checkpoint_dir(ckpt_root)
            require(latest_ckpt is not None, f"missing global_step checkpoint for {variant}: {ckpt_root}")
            require(
                checkpoint_step(latest_ckpt) == expected_step,
                f"{variant} latest checkpoint step mismatch: {latest_ckpt}, expected global_step_{expected_step}",
            )
            require((latest_ckpt / "actor").exists(), f"{variant} checkpoint missing actor dir: {latest_ckpt}")
            require((latest_ckpt / "data.pt").exists(), f"{variant} checkpoint missing dataloader state: {latest_ckpt}")

    print("phase_report_ready_ok")


if __name__ == "__main__":
    main()
