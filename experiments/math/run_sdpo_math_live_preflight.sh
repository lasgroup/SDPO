#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/math_env.sh"

export PHASE="${PHASE:-pilot}"
export VARIANTS="${VARIANTS:-base_model}"
export TRAIN_STEPS="${TRAIN_STEPS:-1}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-64}"
export BASE_MODEL_TRAIN_MAX_SAMPLES="${BASE_MODEL_TRAIN_MAX_SAMPLES:-64}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
export EVAL_FREQ="${EVAL_FREQ:-1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export VERIFY_PHASE_MODEL="${VERIFY_PHASE_MODEL:-0}"
export RUN_TAG="${RUN_TAG:-preflight_live}"
export EXP_SUFFIX="${EXP_SUFFIX:-preflight_live_seed${SEED:-42}}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sdpo_math_phase/preflight_live}"

bash "${SCRIPT_DIR}/run_sdpo_math_benchmark.sh"
