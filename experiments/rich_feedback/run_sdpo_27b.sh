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

ACCOUNT="aira_ws2" # "agentic-models"
QOS="h200_coding_shared" # "h200_agentic-models_high"

DATA_PATHS=(
    "lcb_v6"
)

# Slurm resources - scaled for 27B model
NODES=2
TIME="168:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Sweep Parameters - reduced for 27B memory requirements
TRAIN_BATCH_SIZES=(16)
ROLLOUT_BATCH_SIZES=(4)
MINI_BATCH_SIZES=(4)
LRS=(5e-7)

# SDPO-specific parameters
# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
SEEDS=(42 123 456)

MODEL_PATHS=(
    "Qwen/Qwen3.5-27B"
)

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
                            for SEED in "${SEEDS[@]}"; do
                                for DATA_PATH in "${DATA_PATHS[@]}"; do
                                    # 1. Construct the experiment name (must be unique)
                                    MODEL_NAME="${MODEL_PATH##*/}"
                                    EXP_NAME="FINAL-SDPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-seed${SEED}-model${MODEL_NAME}"

                                    # 2. Construct the arguments string to pass to the training script
                                    # Format: key=value key2=value2 ...
                                    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-rich-feedback-27B \
trainer.nnodes=2 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.data_loader_seed=$SEED \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01"

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

