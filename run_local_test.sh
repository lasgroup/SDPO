#!/bin/bash

# Usage: ./run_local_test.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="sdpo"
# DATA_PATH="datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
# DATA_PATH="datasets/new/gsm8k"
DATA_PATH="datasets/new/math500"

# Hyperparameters (taking the first value from the arrays in run_sdpo.sh)
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
LAMBDA=0.0
CLIP_ADV_HIGH=null
DONTS_REPROMPT_ON_SELF_SUCCESS=True
export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_test"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# Optional: Run setup commands if needed (uncomment if necessary)
# pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0
# pip install -e $PROJECT_ROOT
# pip install --upgrade wandb

# =============================================================================
# EXECUTION
# =============================================================================

# 1. Construct the experiment name
EXP_NAME="LOCAL-SDPO-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip_adv_high${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-${SUFFIX}"

# 2. Construct the arguments string to pass to the training script
# Format: key=value key2=value2 ...
ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.model.path=Qwen/Qwen3-8B \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=1.0 \
actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "Starting Local Training"
echo "Experiment: $EXP_NAME"
echo "Config: $CONFIG_NAME"
echo "Data: $DATA_PATH"
echo "----------------------------------------------------------------"

# 3. Run the training script directly
bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
