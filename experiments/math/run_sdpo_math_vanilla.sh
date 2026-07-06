#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/common_quiet_env.sh"

CONFIG_NAME="${CONFIG_NAME:-sdpo_math_a100}"
MODEL_PATH="${MODEL_PATH:-${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}}"
EXP_NAME="${EXP_NAME:-sdpo_math_vanilla_a100}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
LOGGER="${LOGGER:-[\"console\"]}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-32}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-1}"

if [[ ! -f "${PROJECT_ROOT}/data/dapo_math_en/train.parquet" ]]; then
  echo "Missing data/dapo_math_en/train.parquet. Run Stage 1/2 preprocessing first." >&2
  exit 1
fi

python3 -m verl.trainer.main_ppo \
  --config-name "${CONFIG_NAME}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  critic.model.path="${MODEL_PATH}" \
  critic.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  critic.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.group_name="SDPO-Math-Vanilla" \
  trainer.logger="${LOGGER}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
  data.val_max_samples="${VAL_MAX_SAMPLES}" \
  actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
  actor_rollout_ref.actor.self_distillation.reliability_weighting=False \
  "$@"
