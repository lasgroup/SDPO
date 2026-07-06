# SDPO-Math Phase Runbook

Notebook commands for final SDPO-Math thesis runs.

## Defaults

| Item | Value |
|---|---|
| Python | 3.12 |
| Model | `Qwen/Qwen3-8B` |
| Variants | `base_rl sdpo_vanilla sdpo_reliability_gate` |
| Profile | `quality` |
| Rollout TP | 2 |
| Rollout quantization | `null` |
| Attention | SDPA |
| LoRA | enabled for trained variants |
| Qwen3 response-only logits | true |
| Reliability gate | reliability-weighted sparse SDPO |

| Run | Hardware | Train steps | Train max | Val max |
|---|---|---:|---:|---:|
| Thesis A100/H100 | A100/H100 | 10 | 1024 | 128 |
| Thesis H200 | H200 | 15 | 1536 | 128 |

## Setup

%%bash
set -euo pipefail

cd /root/SDPO
git pull
chmod +x experiments/math/*.sh experiments/math/*.py
unset PYTHON_VERSION
export SDPO_PYTHON_VERSION=3.12
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
bash experiments/math/setup_math_notebook.sh

## Thesis A100/H100

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-10}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Thesis H200

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

export PHASE=thesis
export HARDWARE_PROFILE=h200
export VARIANTS="${VARIANTS:-base_rl sdpo_vanilla sdpo_reliability_gate}"
export TRAIN_STEPS="${TRAIN_STEPS:-15}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1536}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
export ULTRA_QUIET="${ULTRA_QUIET:-1}"
export PROGRESS_WATCH="${PROGRESS_WATCH:-1}"

bash experiments/math/run_sdpo_math_benchmark.sh

## Collect

%%bash
set -euo pipefail

cd /root/SDPO
source experiments/math/math_env.sh

if [[ -z "${LOG_DIR:-}" && -f logs/sdpo_math_phase/latest_thesis_log_dir.txt ]]; then
  LOG_DIR="$(< logs/sdpo_math_phase/latest_thesis_log_dir.txt)"
fi
LOG_DIR="${LOG_DIR:-$(ls -td logs/sdpo_math_phase/* | head -1)}"

python experiments/math/summarize_phase_results.py --log-dir "$LOG_DIR"
python experiments/math/check_phase_report_ready.py \
  --log-dir "$LOG_DIR" \
  --require-checkpoints \
  --expect-phase thesis \
  --expect-model "$THESIS_MODEL_PATH" \
  --expect-profile quality \
  --expect-seed 42
python experiments/math/download_phase_artifacts.py \
  --log-dir "$LOG_DIR" \
  --include-checkpoints \
  --require-checkpoints
cat "$LOG_DIR/manifest.json"
cat "$LOG_DIR/summary.md"
