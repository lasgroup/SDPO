#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
source "${SCRIPT_DIR}/phase_common.sh"

if [[ ! -f "${PROJECT_ROOT}/data/dapo_math_en/train.parquet" ]]; then
  echo "Missing data/dapo_math_en/train.parquet. Run DAPO-Math preprocessing first." >&2
  exit 1
fi

PHASE="${PHASE:-pilot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
LOGGER="${LOGGER:-[\"console\"]}"
CONFIG_NAME="${CONFIG_NAME:-sdpo_math_a100}"
VARIANTS="${VARIANTS:-}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"
VERIFY_PHASE_MODEL="${VERIFY_PHASE_MODEL:-1}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
RELIABILITY_GATE_THRESHOLD="${RELIABILITY_GATE_THRESHOLD:-0.4}"
RELIABILITY_GATE_MAX_FRACTION="${RELIABILITY_GATE_MAX_FRACTION:-0.5}"
RELIABILITY_GATE_SPARSE_EXECUTION="${RELIABILITY_GATE_SPARSE_EXECUTION:-True}"
SDPO_SPARSE_TARGET_EXECUTION="${SDPO_SPARSE_TARGET_EXECUTION:-True}"

for boolean_name in RELIABILITY_GATE_SPARSE_EXECUTION SDPO_SPARSE_TARGET_EXECUTION; do
  case "${!boolean_name}" in
    True|False)
      ;;
    *)
      echo "${boolean_name} must be True or False." >&2
      exit 1
      ;;
  esac
done

case "${HARDWARE_PROFILE}" in
  a100|h100|h200)
    ;;
  *)
    echo "Unknown HARDWARE_PROFILE=${HARDWARE_PROFILE}. Use a100, h100, or h200." >&2
    exit 1
    ;;
esac

case "${PHASE}" in
  pilot)
    DEFAULT_VARIANTS="base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate"
    RUN_PROFILE=fast
    MODEL_PATH="${MODEL_PATH:-${PILOT_MODEL_PATH:-Qwen/Qwen3-1.7B}}"
    TRAIN_STEPS="${TRAIN_STEPS:-10}"
    if [[ "${HARDWARE_PROFILE}" == "h100" ]]; then
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
    else
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-512}"
    fi
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    ROLLOUT_TP="${ROLLOUT_TP:-2}"
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Pilot}"
    ;;
  scale_decision)
    DEFAULT_VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate"
    RUN_PROFILE=fast
    MODEL_PATH="${MODEL_PATH:-${SCALE_MODEL_PATH:-Qwen/Qwen3-8B}}"
    TRAIN_STEPS="${TRAIN_STEPS:-12}"
    if [[ "${HARDWARE_PROFILE}" == "h100" ]]; then
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
    else
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-256}"
    fi
    VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-64}"
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    ROLLOUT_TP="${ROLLOUT_TP:-2}"
    ROLLOUT_QUANTIZATION=null
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Scale-Decision}"
    ;;
  thesis)
    DEFAULT_VARIANTS="base_rl sdpo_vanilla sdpo_reliability_gate"
    RUN_PROFILE=quality
    MODEL_PATH="${MODEL_PATH:-${THESIS_MODEL_PATH:-Qwen/Qwen3-8B}}"
    if [[ "${HARDWARE_PROFILE}" == "h200" ]]; then
      TRAIN_STEPS="${TRAIN_STEPS:-15}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1536}"
      VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    else
      TRAIN_STEPS="${TRAIN_STEPS:-10}"
      TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1024}"
      VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
    fi
    EVAL_FREQ="${EVAL_FREQ:-${TRAIN_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:-${TRAIN_STEPS}}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
    ROLLOUT_TP="${ROLLOUT_TP:-2}"
    ROLLOUT_QUANTIZATION=null
    GROUP_NAME="${GROUP_NAME:-SDPO-Math-Thesis}"
    ;;
  *)
    echo "Unknown PHASE=${PHASE}. Use pilot, scale_decision, or thesis." >&2
    exit 1
    ;;
esac

VARIANTS="${VARIANTS:-${DEFAULT_VARIANTS}}"

if [[ "${CONFIG_NAME}" != "sdpo_math_a100" ]]; then
  echo "Refusing CONFIG_NAME=${CONFIG_NAME}. SDPO-Math phases must use sdpo_math_a100." >&2
  exit 1
fi

case "${MODEL_PATH}" in
  Qwen/Qwen3-1.7B|Qwen/Qwen3-4B|Qwen/Qwen3-8B)
    ;;
  *)
    echo "Refusing MODEL_PATH=${MODEL_PATH}. SDPO-Math phases are locked to Qwen3 1.7B/4B/8B." >&2
    exit 1
    ;;
esac

export CUDA_VISIBLE_DEVICES LOGGER MODEL_PATH HARDWARE_PROFILE RELIABILITY_GATE_THRESHOLD RELIABILITY_GATE_MAX_FRACTION
export RELIABILITY_GATE_SPARSE_EXECUTION SDPO_SPARSE_TARGET_EXECUTION ROLLOUT_TP ROLLOUT_QUANTIZATION
export TRAIN_MAX_SAMPLES VAL_MAX_SAMPLES SEED

RUN_TAG="${RUN_TAG:-${PHASE}_${HARDWARE_PROFILE}_${RUN_PROFILE}_${TRAIN_STEPS}_$(date +%Y%m%d_%H%M%S)}"
EXP_SUFFIX="${EXP_SUFFIX:-${RUN_TAG}_seed${SEED}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/sdpo_math_phase/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"
if [[ "${PHASE}" == "thesis" && "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${PROJECT_ROOT}/logs/sdpo_math_phase"
  printf "%s\n" "${LOG_DIR}" > "${PROJECT_ROOT}/logs/sdpo_math_phase/latest_thesis_log_dir.txt"
fi

sdpo_math_prepare_phase_run "${RUN_PROFILE}" "${LOG_DIR}"

echo "phase=${PHASE} model=${MODEL_PATH} variants=${VARIANTS} dry_run=${DRY_RUN}"
echo "hardware=${HARDWARE_PROFILE}"
echo "reliability_gate_threshold=${RELIABILITY_GATE_THRESHOLD}"
echo "reliability_gate_max_fraction=${RELIABILITY_GATE_MAX_FRACTION}"
echo "reliability_gate_sparse_execution=${RELIABILITY_GATE_SPARSE_EXECUTION}"
echo "sdpo_sparse_target_execution=${SDPO_SPARSE_TARGET_EXECUTION}"
echo "sdpo_rollout_memory=batched_tokens:${SDPO_BATCHED_TOKENS} max_num_seqs:${SDPO_MAX_NUM_SEQS} gpu_util:${SDPO_GPU_UTIL}"
echo "sdpo_actor_memory=actor_len:${SDPO_ACTOR_LEN} reprompt_len:${SDPO_REPROMPT_LEN} activation_offload:${SDPO_ACTIVATION_OFFLOAD} topk:${SDPO_DISTILLATION_TOPK}"
echo "steps=${TRAIN_STEPS} train_max=${TRAIN_MAX_SAMPLES} val_max=${VAL_MAX_SAMPLES} eval_freq=${EVAL_FREQ} save_freq=${SAVE_FREQ} seed=${SEED}"
echo "exp_suffix=${EXP_SUFFIX}"
echo "logs=${LOG_DIR}"

if [[ "${DRY_RUN}" != "1" && "${VERIFY_PHASE_MODEL}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/verify_hf_models.py" --models "${MODEL_PATH}"
fi

python3 "${SCRIPT_DIR}/write_phase_manifest.py" \
  --output "${LOG_DIR}/manifest.json" \
  --config-name "${CONFIG_NAME}" \
  --phase "${PHASE}" \
  --profile "${RUN_PROFILE}" \
  --model "${MODEL_PATH}" \
  --variants "${VARIANTS}" \
  --train-steps "${TRAIN_STEPS}" \
  --train-max-samples "${TRAIN_MAX_SAMPLES}" \
  --val-max-samples "${VAL_MAX_SAMPLES}" \
  --eval-freq "${EVAL_FREQ}" \
  --save-freq "${SAVE_FREQ}" \
  --seed "${SEED}" \
  --exp-suffix "${EXP_SUFFIX}" \
  --log-dir "${LOG_DIR}"

run_with_log() {
  local exp_name="$1"
  shift
  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf "DRY_RUN command:"
      printf " %q" "$@"
      printf "\n"
    } | tee "${LOG_DIR}/${exp_name}.log"
    return 0
  fi
  local progress_pid=""
  local progress_total_steps="${PROGRESS_TOTAL_STEPS:-${TRAIN_STEPS}}"
  cleanup_interrupted_run() {
    local signal_name="$1"
    trap - INT TERM
    echo
    echo "interrupted signal=${signal_name}; stopping progress watcher and Ray"
    if [[ -n "${progress_pid}" ]]; then
      kill "${progress_pid}" >/dev/null 2>&1 || true
      wait "${progress_pid}" >/dev/null 2>&1 || true
    fi
    ray stop --force >/dev/null 2>&1 || true
    return 130
  }
  if [[ "${PROGRESS_WATCH:-1}" == "1" ]]; then
    python3 "${SCRIPT_DIR}/watch_phase_progress.py" \
      --log-dir "${LOG_DIR}" \
      --experiment-name "${exp_name}" \
      --total-steps "${progress_total_steps}" \
      --interval "${PROGRESS_INTERVAL:-15}" &
    progress_pid="$!"
  fi
  local command_start_epoch
  command_start_epoch="$(date +%s)"
  trap 'cleanup_interrupted_run INT; return 130' INT
  trap 'cleanup_interrupted_run TERM; return 143' TERM
  ray stop --force >/dev/null 2>&1 || true
  set +e
  "$@" 2>&1 | tee "${LOG_DIR}/${exp_name}.log"
  local command_status="${PIPESTATUS[0]}"
  set -e
  trap - INT TERM
  if [[ -n "${progress_pid}" ]]; then
    kill "${progress_pid}" >/dev/null 2>&1 || true
    wait "${progress_pid}" >/dev/null 2>&1 || true
  fi
  if [[ "${command_status}" != "0" && "${FAILURE_CONTEXT:-1}" == "1" ]]; then
    python3 "${SCRIPT_DIR}/print_failure_context.py" \
      --variant-log "${LOG_DIR}/${exp_name}.log" \
      --since-epoch "${command_start_epoch}" || true
  fi
  ray stop --force >/dev/null 2>&1 || true
  sleep "${VARIANT_CLEANUP_WAIT_SECONDS:-3}"
  return "${command_status}"
}

run_base_model_val() {
  local exp_name="$1"
  shift
  local base_model_train_max_samples="${BASE_MODEL_TRAIN_MAX_SAMPLES:-$((TRAIN_BS * 2))}"
  PROGRESS_TOTAL_STEPS=0 run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.val_before_train=True \
      trainer.val_only=True \
      trainer.save_freq=-1 \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
      data.train_max_samples="${base_model_train_max_samples}" \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.model.lora_rank=0 \
      actor_rollout_ref.model.lora_alpha=16 \
      actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
      actor_rollout_ref.actor.self_distillation.reliability_weighting=False \
      actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0 \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "${BASE_ROLLOUT_OVERRIDES[@]}" \
      "$@"
}

run_base_rl() {
  local exp_name="$1"
  shift
  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
      data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
      actor_rollout_ref.actor.self_distillation.reliability_weighting=False \
      actor_rollout_ref.actor.self_distillation.reliability_gate_threshold=0.0 \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "${BASE_ROLLOUT_OVERRIDES[@]}" \
      "$@"
}

run_sdpo_variant() {
  local variant="$1"
  local exp_name="$2"
  shift 2
  local include_feedback=True
  local reliability=False
  local reliability_gate_threshold=0.0
  local reliability_gate_max_fraction=null
  local reliability_gate_sparse_execution=False

  case "${variant}" in
    sdpo_vanilla)
      ;;
    sdpo_reliability)
      reliability=True
      ;;
    sdpo_reliability_gate)
      reliability=True
      reliability_gate_threshold="${RELIABILITY_GATE_THRESHOLD}"
      reliability_gate_max_fraction="${RELIABILITY_GATE_MAX_FRACTION}"
      reliability_gate_sparse_execution="${RELIABILITY_GATE_SPARSE_EXECUTION}"
      ;;
    *)
      echo "Unknown SDPO variant=${variant}" >&2
      exit 1
      ;;
  esac

  run_with_log "${exp_name}" \
    python3 -m verl.trainer.main_ppo \
      --config-name "${CONFIG_NAME}" \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      critic.model.path="${MODEL_PATH}" \
      trainer.experiment_name="${exp_name}" \
      trainer.group_name="${GROUP_NAME}" \
      trainer.logger="${LOGGER}" \
      trainer.total_training_steps="${TRAIN_STEPS}" \
      trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
      trainer.test_freq="${EVAL_FREQ}" \
      trainer.save_freq="${SAVE_FREQ}" \
      trainer.validation_data_dir="${LOG_DIR}/validation/${exp_name}" \
      data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
      data.val_max_samples="${VAL_MAX_SAMPLES}" \
      actor_rollout_ref.actor.policy_loss.loss_mode=sdpo \
      actor_rollout_ref.actor.self_distillation.include_environment_feedback="${include_feedback}" \
      actor_rollout_ref.actor.self_distillation.sparse_target_execution="${SDPO_SPARSE_TARGET_EXECUTION}" \
      actor_rollout_ref.actor.self_distillation.reliability_weighting="${reliability}" \
      actor_rollout_ref.actor.self_distillation.reliability_gate_threshold="${reliability_gate_threshold}" \
      actor_rollout_ref.actor.self_distillation.reliability_gate_max_fraction="${reliability_gate_max_fraction}" \
      actor_rollout_ref.actor.self_distillation.reliability_gate_sparse_execution="${reliability_gate_sparse_execution}" \
      "${RAY_LOG_TO_DRIVER_OVERRIDE[@]}" \
      "${COMMON_OVERRIDES[@]}" \
      "${SDPO_ROLLOUT_OVERRIDES[@]}" \
      "$@"
}

for variant in ${VARIANTS}; do
  exp_name="${variant}_${EXP_SUFFIX}"
  echo
  echo "== variant=${variant} exp=${exp_name} =="
  case "${variant}" in
    base_model)
      run_base_model_val "${exp_name}" "$@"
      ;;
    base_rl)
      run_base_rl "${exp_name}" "$@"
      ;;
    sdpo_vanilla|sdpo_reliability|sdpo_reliability_gate)
      run_sdpo_variant "${variant}" "${exp_name}" "$@"
      ;;
    *)
      echo "Unknown variant=${variant}. Valid: base_model base_rl sdpo_vanilla sdpo_reliability sdpo_reliability_gate." >&2
      exit 1
      ;;
  esac
done

echo
echo "done logs=${LOG_DIR}"
