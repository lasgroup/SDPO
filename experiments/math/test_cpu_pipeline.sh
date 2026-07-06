#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON:-python3}"

echo "[1/5] Checking shell script syntax"
bash -n \
  experiments/math/math_env.sh \
  experiments/math/setup_math_notebook.sh \
  experiments/math/run_sdpo_math_benchmark.sh \
  experiments/math/run_sdpo_math_live_preflight.sh \
  experiments/math/run_sdpo_math_vanilla.sh \
  experiments/math/run_sdpo_math_reliability.sh \
  experiments/math/run_sdpo_math_smoke.sh
"${PYTHON_BIN}" -m py_compile \
  experiments/math/preflight_phase.py \
  experiments/math/print_failure_context.py \
  experiments/math/check_phase_report_ready.py \
  experiments/math/inspect_phase_logs.py \
  experiments/math/summarize_phase_results.py \
  experiments/math/verify_hf_models.py \
  experiments/math/watch_phase_progress.py \
  experiments/math/write_phase_manifest.py \
  experiments/math/download_phase_artifacts.py \
  experiments/math/validate_benchmark_dryrun.py
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

expected_defaults = {
    "experiments/math/run_sdpo_math_smoke.sh": "Qwen/Qwen3-1.7B",
    "experiments/math/run_sdpo_math_vanilla.sh": "Qwen/Qwen3-8B",
    "experiments/math/run_sdpo_math_reliability.sh": "Qwen/Qwen3-8B",
}
for path, expected in expected_defaults.items():
    text = Path(path).read_text(encoding="utf-8")
    assert expected in text, f"{path} missing default {expected}"

runner = Path("experiments/math/run_sdpo_math_benchmark.sh").read_text(encoding="utf-8")
for expected in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
    assert expected in runner, f"benchmark runner missing model default {expected}"
math_env = Path("experiments/math/math_env.sh").read_text(encoding="utf-8")
assert 'SCALE_MODEL_PATH="${SCALE_MODEL_PATH:-Qwen/Qwen3-8B}"' in math_env
assert 'MODEL_PATH="${MODEL_PATH:-${SCALE_MODEL_PATH:-Qwen/Qwen3-8B}}"' in runner
for snippet in [
    'VARIANTS="${VARIANTS:-}"',
    'DEFAULT_VARIANTS="base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate"',
    'DEFAULT_VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate"',
    'VARIANTS="${VARIANTS:-${DEFAULT_VARIANTS}}"',
    'TRAIN_STEPS="${TRAIN_STEPS:-12}"',
    'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"',
    'TRAIN_STEPS="${TRAIN_STEPS:-10}"',
    'TRAIN_STEPS="${TRAIN_STEPS:-15}"',
    'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"',
    'TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1536}"',
    'EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"',
    'SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"',
    'RELIABILITY_GATE_THRESHOLD="${RELIABILITY_GATE_THRESHOLD:-0.4}"',
    'RELIABILITY_GATE_SPARSE_EXECUTION="${RELIABILITY_GATE_SPARSE_EXECUTION:-True}"',
    'ROLLOUT_TP="${ROLLOUT_TP:-2}"',
    "ROLLOUT_QUANTIZATION=null",
    'actor_rollout_ref.actor.self_distillation.reliability_gate_threshold="${reliability_gate_threshold}"',
    'actor_rollout_ref.actor.self_distillation.reliability_gate_max_fraction="${reliability_gate_max_fraction}"',
    'actor_rollout_ref.actor.self_distillation.reliability_gate_sparse_execution="${reliability_gate_sparse_execution}"',
]:
    assert snippet in runner, f"benchmark runner missing gate/default logic: {snippet}"

manifest = Path("experiments/math/write_phase_manifest.py").read_text(encoding="utf-8")
for snippet in [
    "variant_hyperparameters",
    "sdpo_reliability",
    "sdpo_reliability_gate",
    "RELIABILITY_GATE_THRESHOLD",
    "ROLLOUT_QUANTIZATION",
    "ROLLOUT_TP",
    "reliability_gate_sparse_execution",
]:
    assert snippet in manifest, f"manifest writer missing gate hyperparameter: {snippet}"

main_ppo = Path("verl/trainer/main_ppo.py").read_text(encoding="utf-8")
for snippet in [
    "def write_progress_heartbeat",
    'write_progress_heartbeat(config, "ray_init_start")',
    'write_progress_heartbeat(config, "task_start")',
    'write_progress_heartbeat(config, "init_workers_start")',
    'write_progress_heartbeat(config, "fit_start")',
]:
    assert snippet in main_ppo, f"main_ppo missing startup progress heartbeat: {snippet}"

trainer = Path("verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
for snippet in [
    "def _progress_heartbeat",
    "VERL_FILE_LOGGER_ROOT",
    ".progress.jsonl",
    'self._progress_heartbeat("resource_pool_start")',
    'self._progress_heartbeat("worker_spawn_start")',
    'self._progress_heartbeat("actor_model_init_start")',
    'self._progress_heartbeat("agent_loop_init_start")',
    'self._progress_heartbeat("step_start")',
    'self._progress_heartbeat("gen_start")',
    'self._progress_heartbeat("actor_update_done")',
    "build_reliability_gate_schedule",
    "_prepare_sparse_self_distillation_actor_batch",
    "self_distillation_sparse_compute_mask",
]:
    assert snippet in trainer, f"trainer missing progress heartbeat: {snippet}"

actor_worker = Path("verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
fsdp_worker = Path("verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
for snippet in [
    "initialize_ema_teacher",
    "_trainable_teacher_parameter_pairs",
    "ema_teacher_update",
    "response_only_logits_kwargs",
    '"logits_to_keep": response_length + 1',
]:
    assert snippet in actor_worker, f"actor worker missing optimized EMA logic: {snippet}"
assert "self.actor.initialize_ema_teacher()" in fsdp_worker
teacher_init = fsdp_worker.index("self.actor.initialize_ema_teacher()")
rollout_init = fsdp_worker.index("self._build_rollout(", teacher_init)
assert teacher_init < rollout_init, "SDPO teacher must initialize before vLLM reserves GPU memory"

watcher = Path("experiments/math/watch_phase_progress.py").read_text(encoding="utf-8")
for snippet in [
    "def progress_path",
    "waiting_for_progress",
    "stage=",
    '"timing_s/gen": "gen_s"',
    '"timing_s/old_log_prob": "oldlp_s"',
    '"response_length/mean": "resp_tok"',
    '"self_distillation/reliability_gate_compute_fraction": "gate_compute"',
    '"timing_s/ema_teacher_update": "ema_s"',
    "read_jsonl_from",
]:
    assert snippet in watcher, f"watcher missing progress heartbeat support: {snippet}"

summary = Path("experiments/math/summarize_phase_results.py").read_text(encoding="utf-8")
for snippet in [
    "time_per_step_s",
    "old_log_prob_s",
    "response_length_mean",
    "response_length_clip_ratio",
    "sdpo_reliability_gate_compute_fraction",
    "sdpo_reliability_gate_compute_token_fraction",
    "ema_teacher_update_s",
    'data.get("timing_s/update_actor", "")',
    "sorted(VARIANTS, key=len, reverse=True)",
]:
    assert snippet in summary, f"summary missing timing field: {snippet}"

import importlib.util

summary_spec = importlib.util.spec_from_file_location("summarize_phase_results", "experiments/math/summarize_phase_results.py")
summary_mod = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary_mod)
assert summary_mod.infer_variant(Path("sdpo_reliability_gate_phase_seed42.jsonl")) == "sdpo_reliability_gate"
assert summary_mod.infer_variant(Path("sdpo_reliability_phase_seed42.jsonl")) == "sdpo_reliability"

if importlib.util.find_spec("torch") and importlib.util.find_spec("ray"):
    import torch
    from torch import nn

    from verl.trainer.ppo.ray_trainer import apply_reliability_gate_budget, build_reliability_gate_schedule
    from verl.workers.actor.dp_actor import response_only_logits_kwargs

    target = torch.tensor([True, False, True, False, True, False, False, False])
    permutation, compute_mask, selected_per_rank = build_reliability_gate_schedule(target, dp_size=2)
    aligned_target = target[permutation]
    assert selected_per_rank == [2, 1]
    assert compute_mask.tolist() == [True, True, False, False, True, True, False, False]
    assert not torch.any(aligned_target & ~compute_mask)
    assert sorted(permutation.tolist()) == list(range(len(target)))
    print("reliability_gate_schedule_ok")

    weights = torch.tensor([1.0, 0.4, 0.4, 0.2, 1.0, 0.4, 0.0, 0.4])
    eligible = weights >= 0.4
    budgeted = apply_reliability_gate_budget(eligible, weights, max_fraction=0.5)
    assert budgeted.sum().item() == 4
    assert budgeted[0] and budgeted[4]
    assert not torch.any(budgeted & ~eligible)
    print("reliability_gate_budget_ok")

    class DummyQwen3(nn.Module):
        config = type("Config", (), {"model_type": "qwen3"})()

    class DummyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.module = DummyQwen3()

    kwargs = response_only_logits_kwargs(
        DummyWrapper(),
        1024,
        enabled=True,
        use_remove_padding=False,
        use_fused_kernels=False,
    )
    assert kwargs == {"logits_to_keep": 1025}
    print("response_only_logits_ok")
else:
    print("reliability_gate_schedule_skipped: torch/ray unavailable")

download_script = Path("experiments/math/download_phase_artifacts.py").read_text(encoding="utf-8")
for snippet in [
    "latest_thesis_log_dir.txt",
    "include-checkpoints",
    "require-checkpoints",
    "latest_checkpointed_iteration.txt",
    "TRAINED_VARIANTS",
]:
    assert snippet in download_script, f"download script missing artifact logic: {snippet}"

phase_common = Path("experiments/math/phase_common.sh").read_text(encoding="utf-8")

fsdp_utils = Path("verl/utils/fsdp_utils.py").read_text(encoding="utf-8")
assert "def collect_lora_and_base_params" in fsdp_utils
assert "model.to(orig_dev)" not in fsdp_utils

fsdp_workers = Path("verl/workers/fsdp_workers.py").read_text(encoding="utf-8")
assert "params, base_model_params = collect_lora_and_base_params" in fsdp_workers
assert "+ray_kwargs.ray_init.log_to_driver=False" in phase_common
assert "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=" in phase_common
assert "\n      ray_kwargs.ray_init.log_to_driver=False" not in phase_common
for snippet in [
    "a100:fast)",
    "a100:balanced)",
    "a100:quality)",
    "h100:fast)",
    "h100:balanced)",
    "h100:quality)",
    "h200:fast)",
    "h200:balanced)",
    "h200:quality)",
    "TRAIN_BS=32",
    "TRAIN_BS=64",
    "ROLLOUT_N=2",
    'ROLLOUT_TP="${ROLLOUT_TP:-2}"',
    'AGENT_WORKERS="${AGENT_WORKERS:-8}"',
    'AGENT_WORKERS="${AGENT_WORKERS:-16}"',
    'MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"',
    'MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"',
    'SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-32768}"',
    'SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-131072}"',
    'SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"',
    'SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-64}"',
    'SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.58}"',
    'SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-3072}"',
    'SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-1536}"',
    'SDPO_ACTIVATION_OFFLOAD="${SDPO_ACTIVATION_OFFLOAD:-True}"',
    "actor_rollout_ref.actor.response_only_logits=True",
    'actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"',
    'ENFORCE_EAGER="${ENFORCE_EAGER:-True}"',
    'actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}"',
    'actor_rollout_ref.rollout.max_num_seqs="${SDPO_MAX_NUM_SEQS}"',
    'actor_rollout_ref.model.enable_activation_offload="${SDPO_ACTIVATION_OFFLOAD}"',
    'actor_rollout_ref.actor.self_distillation.max_reprompt_len="${SDPO_REPROMPT_LEN}"',
    'actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"',
    'actor_rollout_ref.rollout.quantization="${ROLLOUT_QUANTIZATION:-null}"',
]:
    assert snippet in phase_common, f"missing H100 profile setting: {snippet}"

quiet_env = Path("experiments/math/common_quiet_env.sh").read_text(encoding="utf-8")
assert 'VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"' in quiet_env
assert 'PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"' in quiet_env
assert "unset RAY_BACKEND_LOG_LEVEL" in quiet_env
assert "export RAY_BACKEND_LOG_LEVEL" not in quiet_env

setup = Path("experiments/math/setup_math_notebook.sh").read_text(encoding="utf-8")
for snippet in [
    'RUN_CPU_CHECK="${RUN_CPU_CHECK:-0}"',
    'VERIFY_HF_MODELS="${VERIFY_HF_MODELS:-0}"',
    'SKIP_INSTALL_IF_READY="${SKIP_INSTALL_IF_READY:-1}"',
    'FORCE_REINSTALL="${FORCE_REINSTALL:-0}"',
    "ensure_uv",
    "uv venv .venv",
    'uv pip install -e ".[vllm]"',
    "skip_dependency_install=1",
]:
    assert snippet in setup, f"setup script missing lightweight default: {snippet}"
for snippet in [
    "python3 -m pip",
    "python -m pip",
    "uv pip install -q -U pip",
    "RUN_TRANSFORMERS_LOAD_SMOKE",
    "--load-smoke-model",
    "RUN_VLLM_LOAD_SMOKE",
    "RUN_STANDALONE_VLLM_LOAD_SMOKE",
    "--vllm-smoke-model",
]:
    assert snippet not in setup, f"setup script should not run standalone model smoke: {snippet}"
PY

echo "[2/5] Checking YAML config"
"${PYTHON_BIN}" - <<'PY'
import yaml

with open("verl/trainer/config/sdpo_math_a100.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

assert cfg["actor_rollout_ref"]["actor"]["policy_loss"]["loss_mode"] == "sdpo"
assert cfg["actor_rollout_ref"]["model"]["path"] == "Qwen/Qwen3-8B"
assert cfg["critic"]["model"]["path"] == "Qwen/Qwen3-8B"
assert cfg["data"]["train_batch_size"] == 24
assert "val_batch_size" not in cfg["data"]
assert cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"] == 8
assert cfg["actor_rollout_ref"]["rollout"]["max_num_batched_tokens"] == 49152
assert cfg["actor_rollout_ref"]["rollout"]["max_num_seqs"] == 64
assert cfg["actor_rollout_ref"]["rollout"]["enforce_eager"] is True
assert cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]["temperature"] == 0.01
assert cfg["actor_rollout_ref"]["model"]["lora_rank"] > 0
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_weighting"] is False
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_gate_threshold"] == 0.0
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_gate_max_fraction"] is None
assert cfg["actor_rollout_ref"]["actor"]["self_distillation"]["reliability_gate_sparse_execution"] is True
assert cfg["actor_rollout_ref"]["actor"]["response_only_logits"] is True
assert cfg["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
assert cfg["actor_rollout_ref"]["actor"]["shuffle"] is False
assert cfg["trainer"]["n_gpus_per_node"] == 2
assert cfg["reward_manager"]["name"] == "naive"
print("config ok")
PY

echo "[3/5] Checking prepared DAPO-Math parquet"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import pyarrow.parquet as pq

from examples.data_preprocess.dapo_math_processed import DEFAULT_PROMPT_SUFFIX

required = [
    Path("data/dapo_math_en/train.parquet"),
    Path("data/dapo_math_en/val.parquet"),
    Path("data/dapo_math_en/removed_duplicates.jsonl"),
    Path("data/dapo_math_en/removed_contaminated.jsonl"),
    Path("reports/decontamination_report.md"),
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"missing files: {missing}"

for split, expected_min_rows in [("train", 1), ("val", 1)]:
    rows = pq.read_table(f"data/dapo_math_en/{split}.parquet").to_pylist()
    assert len(rows) >= expected_min_rows
    assert all(row["data_source"] == "math_dapo" for row in rows)
    assert all(row["reward_model"]["ground_truth"] for row in rows)
    assert all(row["extra_info"].get("feedback_mode") == "safe" for row in rows)
    assert all(
        row["prompt"][0]["content"].endswith(DEFAULT_PROMPT_SUFFIX)
        for row in rows
    )
    assert not any("Answer:" in row["prompt"][0]["content"] for row in rows)
    print(split, len(rows), "rows ok")
PY

echo "[4/5] Checking math feedback behavior with optional math-verify"
"${PYTHON_BIN}" - <<'PY'
import importlib.util

spec = importlib.util.spec_from_file_location("math_feedback", "verl/utils/reward_score/feedback/math.py")
math_feedback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_feedback)

cases = [
    ("correct", r"Reasoning... \boxed{34}", "34", {"feedback_mode": "safe"}, 1.0, ""),
    (
        "wrong_safe",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your boxed final answer was parsed, but it is incorrect. Recheck the reasoning and final calculation.",
    ),
    ("wrong_none", r"Reasoning... \boxed{35}", "34", {"feedback_mode": "none"}, 0.0, ""),
    (
        "wrong_oracle",
        r"Reasoning... \boxed{35}",
        "34",
        {"feedback_mode": "oracle"},
        0.0,
        "Your answer is incorrect. The correct answer is 34.",
    ),
    (
        "bad_format",
        "Reasoning... final answer is 35",
        "34",
        {"feedback_mode": "safe"},
        0.0,
        "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}.",
    ),
    (
        "truncated",
        "Reasoning...",
        "34",
        {"feedback_mode": "safe", "truncated": True},
        0.0,
        "Your response was truncated because it exceeded the maximum length.",
    ),
]

math_verify_available = None
for name, prediction, ground_truth, extra_info, expected_score, expected_feedback in cases:
    result = math_feedback.compute_score(prediction, ground_truth, extra_info)
    assert result["score"] == expected_score, (name, result)
    assert result["feedback"] == expected_feedback, (name, result)
    assert result["math_verify_available"] in (0, 1), (name, result)
    if math_verify_available is None:
        math_verify_available = result["math_verify_available"]
    assert result["math_verify_available"] == math_verify_available, (name, result)
    print(name, "ok")

print("math_verify_available:", math_verify_available)
if math_verify_available:
    symbolic = math_feedback.compute_score(r"Reasoning... \boxed{1+1}", "2", {"feedback_mode": "safe"})
    print("math_verify symbolic smoke:", symbolic["score"], symbolic["pred"])
PY

echo "[5/5] Checking benchmark variant dry-run"
DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=pilot \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_dryrun \
EXP_SUFFIX=cpu_pipeline_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_dryrun" \
  --hardware-profile a100 \
  --profile fast \
  --exp-suffix cpu_pipeline_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=h100 \
PHASE=pilot \
TRAIN_STEPS=1 \
VARIANTS="base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate" \
RUN_TAG=cpu_pipeline_h100_dryrun \
EXP_SUFFIX=cpu_pipeline_h100_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h100_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_h100_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h100_dryrun" \
  --hardware-profile h100 \
  --profile fast \
  --exp-suffix cpu_pipeline_h100_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=scale_decision \
TRAIN_STEPS=1 \
RUN_TAG=cpu_pipeline_phase2_dryrun \
EXP_SUFFIX=cpu_pipeline_phase2_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_phase2_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_phase2_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_phase2_dryrun" \
  --phase scale_decision \
  --hardware-profile a100 \
  --profile fast \
  --exp-suffix cpu_pipeline_phase2_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=a100 \
PHASE=thesis \
TRAIN_STEPS=1 \
RUN_TAG=cpu_pipeline_thesis_dryrun \
EXP_SUFFIX=cpu_pipeline_thesis_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_thesis_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_thesis_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_thesis_dryrun" \
  --phase thesis \
  --hardware-profile a100 \
  --profile quality \
  --exp-suffix cpu_pipeline_thesis_dryrun_seed42

DRY_RUN=1 \
HARDWARE_PROFILE=h200 \
PHASE=thesis \
TRAIN_STEPS=1 \
RUN_TAG=cpu_pipeline_h200_thesis_dryrun \
EXP_SUFFIX=cpu_pipeline_h200_thesis_dryrun_seed42 \
LOG_DIR="${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h200_thesis_dryrun" \
bash experiments/math/run_sdpo_math_benchmark.sh > /tmp/sdpo_math_cpu_pipeline_h200_thesis_dryrun.log

"${PYTHON_BIN}" experiments/math/validate_benchmark_dryrun.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_h200_thesis_dryrun" \
  --phase thesis \
  --hardware-profile h200 \
  --profile quality \
  --exp-suffix cpu_pipeline_h200_thesis_dryrun_seed42

"${PYTHON_BIN}" experiments/math/download_phase_artifacts.py \
  --log-dir "${PROJECT_ROOT}/logs/sdpo_math_phase/cpu_pipeline_thesis_dryrun" \
  --output-dir /tmp/sdpo_math_download_test >/tmp/sdpo_math_download_test.log

test -s "$(awk -F= '/^archive=/{print $2}' /tmp/sdpo_math_download_test.log)"

echo "CPU pipeline checks passed"
