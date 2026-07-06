#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/common_quiet_env.sh"

VARIANT="${1:-vanilla}"
shift || true

case "${VARIANT}" in
  vanilla)
    INCLUDE_FEEDBACK=True
    RELIABILITY_WEIGHTING=False
    EXP_NAME="${EXP_NAME:-sdpo_math_smoke_vanilla}"
    ;;
  reliability)
    INCLUDE_FEEDBACK=True
    RELIABILITY_WEIGHTING=True
    EXP_NAME="${EXP_NAME:-sdpo_math_smoke_reliability}"
    ;;
  *)
    echo "Usage: $0 [vanilla|reliability] [hydra overrides...]" >&2
    exit 1
    ;;
esac

CONFIG_NAME="${CONFIG_NAME:-sdpo_math_a100}"
MODEL_PATH="${MODEL_PATH:-${PILOT_MODEL_PATH:-Qwen/Qwen3-1.7B}}"
LOGGER="${LOGGER:-[\"console\"]}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-2}"
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
  trainer.group_name="SDPO-Math-Smoke" \
  trainer.logger="${LOGGER}" \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
  data.filter_overlong_prompts_workers="${FILTER_OVERLONG_PROMPTS_WORKERS}" \
  data.train_max_samples=8 \
  data.val_max_samples=8 \
  data.train_batch_size=2 \
  data.max_response_length=1024 \
  rollout_model_len=3072 \
  actor_max_token_len=3072 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
  actor_rollout_ref.actor.self_distillation.max_reprompt_len=2048 \
  actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
  actor_rollout_ref.actor.self_distillation.include_environment_feedback="${INCLUDE_FEEDBACK}" \
  actor_rollout_ref.actor.self_distillation.reliability_weighting="${RELIABILITY_WEIGHTING}" \
  "$@"
