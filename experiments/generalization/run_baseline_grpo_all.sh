#!/bin/bash

# Usage: ./run_baseline_grpo_all.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="baseline_grpo"
BASE_JOB_NAME="rlvr"

DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

# Fixed Slurm resources
ACCOUNT="a156"
NODES=1
PARTITION="normal"
TIME="12:00:00"
ENV="sdpo"
NTASKS_PER_NODE=1
GPUS_PER_NODE=4
MEM=460000
CPUS_PER_TASK=288

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(8 32)

LRS=(1e-5 1e-6)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    "allenai/Olmo-3-7B-Instruct"
)

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"
    # Define the environment setup and command execution
    # We use the user's home directory dynamically
    local setup_cmds="pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0; \
pip install -e /users/$USER/SDPO; \
pip install --upgrade wandb; \
export PYTHONPATH=/users/$USER/SDPO:\$PYTHONPATH"

    local run_cmd="bash /users/$USER/SDPO/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --partition="$PARTITION"
        --time="$TIME"
        --environment="$ENV"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="/users/$USER/output/SDPO/%j.log"
        --error="/users/$USER/output/SDPO/%j.err"
        --wrap="$wrapped_cmd"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        echo "Submitting job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do
                        # 1. Construct the experiment name (must be unique)
                        EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}"

                        # 2. Construct the arguments string to pass to the training script
                        # Format: key=value key2=value2 ...
                        ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-generalization \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

                        # 3. Submit
                        submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                    done
                done
            done
        done
    done
done

