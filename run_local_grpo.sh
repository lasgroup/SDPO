#!/bin/bash

# Usage: ./run_local_grpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="baseline_grpo"

# Default to ToolUse dataset
DATA_PATH="datasets/tooluse"

# Hyperparameters (from experiments/run_baseline_grpo_all.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=8
LR=1e-5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_grpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

EXP_NAME="LOCAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_PATH}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-local \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "Starting Local GRPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
