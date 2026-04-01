#!/bin/bash

# Usage: ./run_sdpo.sh [--dry-run]

# Source common utilities
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SOURCE_DIR/common.sh"
parse_dry_run "$1"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="sdpo"
BASE_JOB_NAME="SDPO"

ACCOUNT="agentic-models" # "aira_ws2" # 
QOS= "h200_agentic-models_high" # "h200_coding_shared" # 

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

# SDPO-specific parameters
# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

# Random seeds for reproducibility
SEEDS=(42 123 456)
# =============================================================================
# MAIN SWEEP LOOP
# =============================================================================

for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                            for DATA_PATH in "${DATA_PATHS[@]}"; do
                                for SEED in "${SEEDS[@]}"; do
                                    # 1. Construct the experiment name (must be unique)
                                    EXP_NAME="FINAL-SDPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-model${MODEL_PATH}-seed${SEED}"

                                    # 2. Construct the arguments string to pass to the training script
                                    # Format: key=value key2=value2 ...
                                    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-rich-feedback \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.data_loader_seed=$SEED"

                                    # 3. Submit
                                    submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

