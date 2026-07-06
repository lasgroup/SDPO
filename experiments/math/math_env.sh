#!/usr/bin/env bash

# Source this from notebook cells before running SDPO-Math commands.
# It centralizes repo paths, cache paths, quiet logging defaults, and model defaults.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it: source experiments/math/math_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${SDPO_SKIP_VENV:-0}" != "1" && -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_quiet_env.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.cache/uv}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export SMOKE_MODEL_PATH="${SMOKE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
export SCALE_MODEL_PATH="${SCALE_MODEL_PATH:-Qwen/Qwen3-8B}"
export THESIS_MODEL_PATH="${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}"
export PILOT_MODEL_PATH="${PILOT_MODEL_PATH:-${SMOKE_MODEL_PATH}}"
export TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-${THESIS_MODEL_PATH}}"

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints"

if [[ "${SDPO_ENV_PRINT:-0}" == "1" ]]; then
  echo "repo=${PROJECT_ROOT}"
  echo "python=$(command -v python || true)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "pilot_model=${PILOT_MODEL_PATH}"
  echo "scale_model=${SCALE_MODEL_PATH}"
  echo "thesis_model=${THESIS_MODEL_PATH}"
fi
