#!/usr/bin/env python3
"""Validate SDPO-Math benchmark dry-run command semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMMON_SNIPPETS = {
    "base_model": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path={model}",
        "critic.model.path={model}",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.model.lora_rank=0",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0",
    ],
    "base_rl": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path={model}",
        "critic.model.path={model}",
        "trainer.total_training_steps={train_steps}",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0",
    ],
    "sdpo_vanilla": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path={model}",
        "critic.model.path={model}",
        "trainer.total_training_steps={train_steps}",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.sparse_target_execution=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=False",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_max_fraction=null",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_sparse_execution=False",
    ],
    "sdpo_reliability": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path={model}",
        "critic.model.path={model}",
        "trainer.total_training_steps={train_steps}",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.sparse_target_execution=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=True",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_max_fraction=null",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_sparse_execution=False",
    ],
    "sdpo_reliability_gate": [
        "--config-name sdpo_math_a100",
        "actor_rollout_ref.model.path={model}",
        "critic.model.path={model}",
        "trainer.total_training_steps={train_steps}",
        "trainer.validation_data_dir=",
        "actor_rollout_ref.actor.policy_loss.loss_mode=sdpo",
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback=True",
        "actor_rollout_ref.actor.self_distillation.sparse_target_execution=True",
        "actor_rollout_ref.actor.self_distillation.reliability_weighting=True",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_threshold={reliability_gate_threshold}",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_max_fraction={reliability_gate_max_fraction}",
        "actor_rollout_ref.actor.self_distillation.reliability_gate_sparse_execution=True",
    ],
}

FORBIDDEN_SNIPPETS = [
    "data.val_batch_size=",
    "RAY_BACKEND_LOG_LEVEL",
]

PROFILE_EXPECTATIONS = {
    ("a100", "fast"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.72",
        "sdpo_max_num_seqs": 32,
        "sdpo_gpu_util": "0.58",
        "sdpo_actor_len": 3072,
        "sdpo_reprompt_len": 1536,
        "enforce_eager": "True",
    },
    ("a100", "balanced"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.72",
        "sdpo_max_num_seqs": 32,
        "sdpo_gpu_util": "0.56",
        "sdpo_actor_len": 4096,
        "sdpo_reprompt_len": 2048,
        "enforce_eager": "True",
    },
    ("a100", "quality"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.70",
        "sdpo_max_num_seqs": 32,
        "sdpo_gpu_util": "0.54",
        "sdpo_actor_len": 6144,
        "sdpo_reprompt_len": 3072,
        "enforce_eager": "True",
    },
    ("h100", "fast"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.92",
        "sdpo_max_num_seqs": 48,
        "sdpo_gpu_util": "0.78",
        "sdpo_actor_len": 4096,
        "sdpo_reprompt_len": 2048,
        "enforce_eager": "True",
    },
    ("h100", "balanced"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.93",
        "sdpo_max_num_seqs": 48,
        "sdpo_gpu_util": "0.78",
        "sdpo_actor_len": 6144,
        "sdpo_reprompt_len": 3072,
        "enforce_eager": "True",
    },
    ("h100", "quality"): {
        "train_batch_size": 32,
        "agent_workers": 8,
        "base_model_train_max_samples": 64,
        "max_num_seqs": 64,
        "gpu_util": "0.93",
        "sdpo_max_num_seqs": 48,
        "sdpo_gpu_util": "0.76",
        "sdpo_actor_len": 8192,
        "sdpo_reprompt_len": 4096,
        "enforce_eager": "True",
    },
    ("h200", "fast"): {
        "train_batch_size": 64,
        "agent_workers": 16,
        "base_model_train_max_samples": 128,
        "max_num_seqs": 96,
        "gpu_util": "0.70",
        "sdpo_max_num_seqs": 64,
        "sdpo_gpu_util": "0.60",
        "sdpo_actor_len": 4096,
        "sdpo_reprompt_len": 2048,
        "enforce_eager": "True",
    },
    ("h200", "balanced"): {
        "train_batch_size": 64,
        "agent_workers": 16,
        "base_model_train_max_samples": 128,
        "max_num_seqs": 96,
        "gpu_util": "0.70",
        "sdpo_max_num_seqs": 64,
        "sdpo_gpu_util": "0.60",
        "sdpo_actor_len": 6144,
        "sdpo_reprompt_len": 3072,
        "enforce_eager": "True",
    },
    ("h200", "quality"): {
        "train_batch_size": 64,
        "agent_workers": 16,
        "base_model_train_max_samples": 128,
        "max_num_seqs": 96,
        "gpu_util": "0.70",
        "sdpo_max_num_seqs": 64,
        "sdpo_gpu_util": "0.58",
        "sdpo_actor_len": 8192,
        "sdpo_reprompt_len": 4096,
        "enforce_eager": "True",
    },
}


def load_manifest(log_dir: Path) -> dict | None:
    manifest_path = log_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def manifest_variants(manifest: dict | None) -> list[str] | None:
    if manifest is None:
        return None
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not all(isinstance(item, str) for item in variants):
        raise AssertionError(f"manifest has invalid variants: {variants}")
    return variants


def manifest_gate_threshold(manifest: dict | None) -> str:
    if manifest is None:
        return "0.4"
    gate_cfg = manifest.get("variant_hyperparameters", {}).get("sdpo_reliability_gate", {})
    threshold = gate_cfg.get("reliability_gate_threshold", "0.4")
    return str(threshold)


def manifest_gate_max_fraction(manifest: dict | None) -> str:
    if manifest is None:
        return "0.5"
    gate_cfg = manifest.get("variant_hyperparameters", {}).get("sdpo_reliability_gate", {})
    max_fraction = gate_cfg.get("reliability_gate_max_fraction", "0.5")
    return str(max_fraction)


def manifest_model(manifest: dict | None) -> str:
    if manifest is None:
        return "Qwen/Qwen3-1.7B"
    return str(manifest.get("model", "Qwen/Qwen3-1.7B"))


def manifest_train_steps(manifest: dict | None, fallback: str) -> str:
    if manifest is None:
        return fallback
    return str(manifest.get("train_steps", fallback))


def manifest_rollout_quantization(manifest: dict | None) -> str:
    if manifest is None:
        return "null"
    profile_settings = manifest.get("profile_settings") or {}
    return str(profile_settings.get("rollout_quantization") or "null")


def manifest_rollout_tp(manifest: dict | None, phase: str) -> str:
    if manifest is None:
        return "2"
    profile_settings = manifest.get("profile_settings") or {}
    return str(profile_settings.get("rollout_tp") or "2")


def manifest_agent_workers(manifest: dict | None, fallback: int) -> str:
    if manifest is None:
        return str(fallback)
    profile_settings = manifest.get("profile_settings") or {}
    return str(profile_settings.get("agent_workers") or fallback)


def expected_snippets(
    hardware_profile: str,
    profile: str,
    variants: list[str],
    reliability_gate_threshold: str,
    reliability_gate_max_fraction: str,
    model: str,
    train_steps: str,
    rollout_tp: str,
    agent_workers: str,
    rollout_quantization: str,
) -> dict[str, list[str]]:
    settings = PROFILE_EXPECTATIONS.get((hardware_profile, profile))
    if settings is None:
        raise AssertionError(
            "validate_benchmark_dryrun.py does not know "
            f"hardware_profile={hardware_profile} profile={profile}. "
            f"Known combinations: {sorted(PROFILE_EXPECTATIONS)}"
        )

    result = {}
    for variant in variants:
        if variant not in COMMON_SNIPPETS:
            raise AssertionError(f"unknown dry-run variant={variant}. Known variants: {sorted(COMMON_SNIPPETS)}")
        snippets = [
            snippet.format(
                reliability_gate_threshold=reliability_gate_threshold,
                reliability_gate_max_fraction=reliability_gate_max_fraction,
                model=model,
                train_steps=train_steps,
            )
            for snippet in COMMON_SNIPPETS[variant]
        ]
        result[variant] = snippets
    if "base_model" in result:
        result["base_model"].append(f"data.train_max_samples={settings['base_model_train_max_samples']}")
    for variant, snippets in result.items():
        max_num_seqs = settings["sdpo_max_num_seqs"] if variant.startswith("sdpo_") else settings["max_num_seqs"]
        gpu_util = settings["sdpo_gpu_util"] if variant.startswith("sdpo_") else settings["gpu_util"]
        actor_len = settings["sdpo_actor_len"] if variant.startswith("sdpo_") else None
        reprompt_len = settings["sdpo_reprompt_len"] if variant.startswith("sdpo_") else None
        snippets.append(f"data.train_batch_size={settings['train_batch_size']}")
        snippets.append("actor_rollout_ref.actor.response_only_logits=True")
        snippets.append(f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}")
        snippets.append(f"actor_rollout_ref.rollout.agent.num_workers={agent_workers}")
        snippets.append(f"actor_rollout_ref.rollout.max_num_seqs={max_num_seqs}")
        snippets.append(f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_util}")
        if actor_len is not None:
            snippets.append(f"actor_max_token_len={actor_len}")
            snippets.append(f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor_len}")
            snippets.append("actor_rollout_ref.model.enable_activation_offload=True")
        if reprompt_len is not None:
            snippets.append(f"actor_rollout_ref.actor.self_distillation.max_reprompt_len={reprompt_len}")
        snippets.append(f"actor_rollout_ref.rollout.enforce_eager={settings['enforce_eager']}")
        snippets.append(f"actor_rollout_ref.rollout.quantization={rollout_quantization}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--phase", default="pilot")
    parser.add_argument("--hardware-profile", default="a100")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--steps", default="1")
    parser.add_argument("--exp-suffix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing_files: list[str] = []
    missing_snippets: list[str] = []
    exp_suffix = args.exp_suffix or f"{args.phase}_{args.profile}_{args.steps}_seed42"
    manifest = load_manifest(args.log_dir)
    variants = manifest_variants(manifest) or list(COMMON_SNIPPETS)
    reliability_gate_threshold = manifest_gate_threshold(manifest)
    reliability_gate_max_fraction = manifest_gate_max_fraction(manifest)
    model = manifest_model(manifest)
    train_steps = manifest_train_steps(manifest, args.steps)
    rollout_tp = manifest_rollout_tp(manifest, args.phase)
    settings = PROFILE_EXPECTATIONS.get((args.hardware_profile, args.profile))
    if settings is None:
        raise AssertionError(
            "validate_benchmark_dryrun.py does not know "
            f"hardware_profile={args.hardware_profile} profile={args.profile}. "
            f"Known combinations: {sorted(PROFILE_EXPECTATIONS)}"
        )
    agent_workers = manifest_agent_workers(manifest, settings["agent_workers"])
    rollout_quantization = manifest_rollout_quantization(manifest)

    for variant, snippets in expected_snippets(
        args.hardware_profile,
        args.profile,
        variants,
        reliability_gate_threshold,
        reliability_gate_max_fraction,
        model,
        train_steps,
        rollout_tp,
        agent_workers,
        rollout_quantization,
    ).items():
        path = args.log_dir / f"{variant}_{exp_suffix}.log"
        if not path.exists():
            missing_files.append(str(path))
            continue
        text = path.read_text(encoding="utf-8")
        if "DRY_RUN command:" not in text:
            missing_snippets.append(f"{path}: missing DRY_RUN command")
        for snippet in snippets:
            if snippet not in text:
                missing_snippets.append(f"{path}: missing {snippet}")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                missing_snippets.append(f"{path}: forbidden {snippet}")

    if missing_files or missing_snippets:
        details = "\n".join(missing_files + missing_snippets)
        raise AssertionError(f"benchmark dry-run validation failed:\n{details}")

    print("benchmark_dryrun_semantics_ok")


if __name__ == "__main__":
    main()
