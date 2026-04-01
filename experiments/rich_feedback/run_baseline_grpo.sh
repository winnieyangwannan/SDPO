#!/bin/bash

# Usage: ./run_baseline_grpo.sh [--dry-run]
# chmod +x /home/winnieyangwn/SDPO/experiments/rich_feedback/run_baseline_grpo.sh
# Source common utilities
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SOURCE_DIR/common.sh"
parse_dry_run "$1"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="baseline_grpo"
BASE_JOB_NAME="GRPO"

ACCOUNT="agentic-models" # "aira_ws2" # "agentic-models"
QOS="h200_agentic-models_high" # "h200_coding_shared" # "h200_agentic-models_high"


DATA_PATHS=(
    "lcb_v6"
)

# Slurm resources
NODES=1
TIME="168:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(8)

LRS=(1e-6)

# Checkpoint saving frequency (-1 to disable, positive number for every N steps)
SAVE_FREQ=10

SEEDS=(42 123 456)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            # 1. Construct the experiment name (must be unique)
                            MODEL_NAME="${MODEL_PATH##*/}"
                            EXP_NAME="FINAL-GRPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-seed${SEED}-model${MODEL_NAME}"

                            # 2. Construct the arguments string to pass to the training script
                            # Format: key=value key2=value2 ...
                            ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=GRPO-rich-feedback \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.data_loader_seed=$SEED \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
trainer.save_freq=$SAVE_FREQ"

                            # 3. Submit
                            submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done

