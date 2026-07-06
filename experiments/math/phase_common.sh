#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_quiet_env.sh"

sdpo_math_configure_profile() {
  local profile="${1:?profile required}"
  local hardware="${HARDWARE_PROFILE:-a100}"

  case "${hardware}" in
    a100|h100|h200)
      ;;
    *)
      echo "Unknown HARDWARE_PROFILE=${hardware}. Use a100, h100, or h200." >&2
      return 1
      ;;
  esac

  case "${hardware}:${profile}" in
    a100:fast)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=1024
      MODEL_LEN=3072
      ACTOR_LEN=4096
      REPROMPT_LEN=2048
      BATCHED_TOKENS="${BATCHED_TOKENS:-49152}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.72}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-32768}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.58}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-3072}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-1536}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    a100:balanced)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=1536
      MODEL_LEN=4096
      ACTOR_LEN=6144
      REPROMPT_LEN=3072
      BATCHED_TOKENS="${BATCHED_TOKENS:-65536}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.72}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-49152}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.56}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-4096}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-2048}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    a100:quality)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=2048
      MODEL_LEN=6144
      ACTOR_LEN=8192
      REPROMPT_LEN=4096
      BATCHED_TOKENS="${BATCHED_TOKENS:-98304}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.70}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-49152}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-32}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.54}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-6144}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-3072}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h100:fast)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=1024
      MODEL_LEN=3072
      ACTOR_LEN=4096
      REPROMPT_LEN=2048
      BATCHED_TOKENS="${BATCHED_TOKENS:-49152}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.92}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-49152}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-48}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.78}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-4096}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-2048}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h100:balanced)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=1536
      MODEL_LEN=4096
      ACTOR_LEN=6144
      REPROMPT_LEN=3072
      BATCHED_TOKENS="${BATCHED_TOKENS:-65536}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.93}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-49152}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-48}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.78}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-6144}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-3072}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h100:quality)
      TRAIN_BS=32
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-8}"
      RESPONSE_LEN=2048
      MODEL_LEN=6144
      ACTOR_LEN=8192
      REPROMPT_LEN=4096
      BATCHED_TOKENS="${BATCHED_TOKENS:-98304}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
      GPU_UTIL="${GPU_UTIL:-0.93}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-65536}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-48}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.76}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-8192}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-4096}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h200:fast)
      TRAIN_BS=64
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-16}"
      RESPONSE_LEN=1024
      MODEL_LEN=3072
      ACTOR_LEN=4096
      REPROMPT_LEN=2048
      BATCHED_TOKENS="${BATCHED_TOKENS:-98304}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
      GPU_UTIL="${GPU_UTIL:-0.70}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-65536}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-64}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.60}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-4096}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-2048}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h200:balanced)
      TRAIN_BS=64
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-16}"
      RESPONSE_LEN=1536
      MODEL_LEN=4096
      ACTOR_LEN=6144
      REPROMPT_LEN=3072
      BATCHED_TOKENS="${BATCHED_TOKENS:-131072}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
      GPU_UTIL="${GPU_UTIL:-0.70}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-98304}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-64}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.60}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-6144}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-3072}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    h200:quality)
      TRAIN_BS=64
      ROLLOUT_N=2
      AGENT_WORKERS="${AGENT_WORKERS:-16}"
      RESPONSE_LEN=2048
      MODEL_LEN=6144
      ACTOR_LEN=8192
      REPROMPT_LEN=4096
      BATCHED_TOKENS="${BATCHED_TOKENS:-196608}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
      GPU_UTIL="${GPU_UTIL:-0.70}"
      SDPO_BATCHED_TOKENS="${SDPO_BATCHED_TOKENS:-131072}"
      SDPO_MAX_NUM_SEQS="${SDPO_MAX_NUM_SEQS:-64}"
      SDPO_GPU_UTIL="${SDPO_GPU_UTIL:-0.58}"
      SDPO_ACTOR_LEN="${SDPO_ACTOR_LEN:-8192}"
      SDPO_REPROMPT_LEN="${SDPO_REPROMPT_LEN:-4096}"
      ENFORCE_EAGER="${ENFORCE_EAGER:-True}"
      ;;
    *)
      echo "Unknown RUN_PROFILE=${profile}. Use fast, balanced, or quality." >&2
      return 1
      ;;
  esac

  ROLLOUT_TP="${ROLLOUT_TP:-2}"
  SDPO_ACTIVATION_OFFLOAD="${SDPO_ACTIVATION_OFFLOAD:-True}"
  SDPO_DISTILLATION_TOPK="${SDPO_DISTILLATION_TOPK:-50}"
  RELIABILITY_GATE_MAX_FRA  CTION="${RELIABILITY_GATE_MAX_FRACTION:-0.5}"

  export TRAIN_BS ROLLOUT_N AGENT_WORKERS RESPONSE_LEN MODEL_LEN ACTOR_LEN REPROMPT_LEN
  export BATCHED_TOKENS MAX_NUM_SEQS GPU_UTIL SDPO_BATCHED_TOKENS SDPO_MAX_NUM_SEQS SDPO_GPU_UTIL
  export SDPO_ACTOR_LEN SDPO_REPROMPT_LEN SDPO_ACTIVATION_OFFLOAD SDPO_DISTILLATION_TOPK
  export ENFORCE_EAGER ROLLOUT_TP RELIABILITY_GATE_MAX_FRACTION
}

sdpo_math_validate_profile() {
  local total_rollouts=$((TRAIN_BS * ROLLOUT_N))
  if (( ROLLOUT_TP < 1 )); then
    echo "Invalid profile: ROLLOUT_TP must be >= 1." >&2
    return 1
  fi
  if (( 2 % ROLLOUT_TP != 0 )); then
    echo "Invalid profile: ROLLOUT_TP must divide the 2-GPU phase shape." >&2
    return 1
  fi
  if (( total_rollouts < AGENT_WORKERS )); then
    echo "Invalid profile: train_batch_size * rollout.n must be >= agent workers." >&2
    return 1
  fi
  if (( total_rollouts % AGENT_WORKERS != 0 )); then
    echo "Invalid profile: train_batch_size * rollout.n must be divisible by agent workers." >&2
    return 1
  fi
}

sdpo_math_init_logging() {
  local log_dir="${1:?log_dir required}"

  export LOGGER="${LOGGER:-[\"console\"]}"
  export VERL_FILE_LOGGER_ROOT="${log_dir}/metrics"
  RAY_LOG_TO_DRIVER_OVERRIDE=(
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED='"1"'
  )

  if [[ "${ULTRA_QUIET:-0}" == "1" ]]; then
    export LOGGER='["file"]'
    RAY_LOG_TO_DRIVER_OVERRIDE+=(+ray_kwargs.ray_init.log_to_driver=False)
    echo "ultra_quiet=1 metrics=${VERL_FILE_LOGGER_ROOT}"
  else
    echo "ultra_quiet=0 logger=${LOGGER} progress=${VERL_FILE_LOGGER_ROOT}"
  fi
}

sdpo_math_build_common_overrides() {
  COMMON_OVERRIDES=(
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.override_config.attn_implementation=sdpa
    critic.model.use_remove_padding=False
    critic.model.override_config.attn_implementation=sdpa
    data.dataloader_num_workers=0
    data.filter_overlong_prompts_workers=1
    data.seed="${SEED:-42}"
    data.train_batch_size="${TRAIN_BS}"
    data.max_response_length="${RESPONSE_LEN}"
    rollout_model_len="${MODEL_LEN}"
    actor_max_token_len="${ACTOR_LEN}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_LEN}"
    actor_rollout_ref.actor.response_only_logits=True
    actor_rollout_ref.actor.data_loader_seed="${SEED:-42}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}"
    actor_rollout_ref.rollout.max_model_len="${MODEL_LEN}"
    actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}"
    actor_rollout_ref.rollout.quantization="${ROLLOUT_QUANTIZATION:-null}"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MODEL_LEN}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0.01
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MODEL_LEN}"
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="${REPROMPT_LEN}"
    critic.data_loader_seed="${SEED:-42}"
  )

  BASE_ROLLOUT_OVERRIDES=(
    actor_rollout_ref.rollout.max_num_batched_tokens="${BATCHED_TOKENS}"
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_UTIL}"
  )

  SDPO_ROLLOUT_OVERRIDES=(
    actor_max_token_len="${SDPO_ACTOR_LEN}"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${SDPO_ACTOR_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${SDPO_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.max_num_seqs="${SDPO_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${SDPO_GPU_UTIL}"
    actor_rollout_ref.model.enable_activation_offload="${SDPO_ACTIVATION_OFFLOAD}"
    actor_rollout_ref.actor.self_distillation.max_reprompt_len="${SDPO_REPROMPT_LEN}"
    actor_rollout_ref.actor.self_distillation.distillation_topk="${SDPO_DISTILLATION_TOPK}"
  )
}

sdpo_math_prepare_phase_run() {
  local profile="${1:?profile required}"
  local log_dir="${2:?log_dir required}"

  sdpo_math_configure_profile "${profile}"
  sdpo_math_validate_profile
  sdpo_math_init_logging "${log_dir}"
  sdpo_math_build_common_overrides

  echo "hardware=${HARDWARE_PROFILE:-a100} profile=${profile} train_bs=${TRAIN_BS} rollout_n=${ROLLOUT_N} rollout_tp=${ROLLOUT_TP} effective_rollouts=$((TRAIN_BS * ROLLOUT_N)) agent_workers=${AGENT_WORKERS} response_len=${RESPONSE_LEN} model_len=${MODEL_LEN} batched_tokens=${BATCHED_TOKENS} max_num_seqs=${MAX_NUM_SEQS} gpu_util=${GPU_UTIL} response_only_logits=True sdpo_batched_tokens=${SDPO_BATCHED_TOKENS} sdpo_max_num_seqs=${SDPO_MAX_NUM_SEQS} sdpo_gpu_util=${SDPO_GPU_UTIL} sdpo_actor_len=${SDPO_ACTOR_LEN} sdpo_reprompt_len=${SDPO_REPROMPT_LEN} sdpo_activation_offload=${SDPO_ACTIVATION_OFFLOAD} reliability_gate_max_fraction=${RELIABILITY_GATE_MAX_FRACTION} enforce_eager=${ENFORCE_EAGER} rollout_quantization=${ROLLOUT_QUANTIZATION:-null}"
}
