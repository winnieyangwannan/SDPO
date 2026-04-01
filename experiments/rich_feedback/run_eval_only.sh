#!/bin/bash

# =============================================================================
# Evaluation-Only Script for LiveCodeBench
# =============================================================================
# Usage: 
#   ./run_eval_only.sh --dry-run                    # Preview commands
#   ./run_eval_only.sh                               # Submit jobs
#
# This script runs evaluation only (no training) on specified model checkpoints.
# It supports two modes:
#   1. HF Model Path: Evaluate a HuggingFace format model directly
#   2. Training Checkpoint: Resume from a training checkpoint path
# =============================================================================

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
BASE_JOB_NAME="EVAL"

ACCOUNT="aira_ws2"
QOS="h200_coding_shared"

# Data path for evaluation
DATA_PATH="lcb_v6"

# Fixed Slurm resources (can use fewer resources for eval-only)
NODES=1
TIME="4:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Number of samples per prompt for evaluation
VAL_N=16

# =============================================================================
# MODEL CHECKPOINTS TO EVALUATE
# =============================================================================
# Option 1: HuggingFace format model paths (already converted)
# Option 2: Training checkpoint paths (will use resume_from_path)
#
# For training checkpoints, use the format:
#   /checkpoint/agentic-models/user/SDPO/GRPO/checkpoints/<exp_name>/global_step_<N>/actor
#
# For HF models (converted or original):
#   /checkpoint/agentic-models/user/models/Qwen3.5-27B
# =============================================================================

# Example: Specify HF model paths to evaluate
HF_MODEL_PATHS=(
    # "/checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-27B"
    # Add more model paths here
)

# Example: Specify training checkpoint paths to evaluate
# The checkpoint path should point to the global_step_X directory
TRAINING_CHECKPOINTS=(
    # "/checkpoint/agentic-models/winnieyangwn/SDPO/GRPO/checkpoints/FINAL-GRPO-mbs-16-train32-rollout8-lr1e-6-seed42-modelQwen3.5-27B/global_step_100"
    # Add more checkpoint paths here
)

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_eval_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Define the environment setup and command execution
    local setup_cmds="eval \"\$(conda shell.bash hook)\"; conda activate verl2; export PYTHONPATH=/home/$USER/SDPO:\$PYTHONPATH; export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export RAY_DISABLE_METRICS=1; export VLLM_ATTENTION_BACKEND=XFORMERS; export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa"

    local run_cmd="bash /home/$USER/SDPO/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$BASE_JOB_NAME"
        --account="$ACCOUNT"
        --nodes="$NODES"
        --qos="$QOS"
        --time="$TIME"
        --ntasks-per-node="$NTASKS_PER_NODE"
        --gpus-per-node="$GPUS_PER_NODE"
        --mem="$MEM"
        --cpus-per-task="$CPUS_PER_TASK"
        --output="/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/%j.log"
        --error="/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/%j.err"
        --wrap="$wrapped_cmd"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit eval job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        # Ensure output directory exists
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs"
        echo "Submitting eval job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}

# =============================================================================
# EVALUATE HF MODEL PATHS
# =============================================================================
for MODEL_PATH in "${HF_MODEL_PATHS[@]}"; do
    MODEL_NAME="${MODEL_PATH##*/}"
    EXP_NAME="EVAL-${MODEL_NAME}"

    # Evaluation-only arguments
    ARGS="trainer.val_only=True \
trainer.val_before_train=True \
trainer.group_name=EVAL-rich-feedback \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
trainer.nnodes=$NODES \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
algorithm.adv_estimator=grpo"

    submit_eval_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
done

# =============================================================================
# EVALUATE TRAINING CHECKPOINTS
# =============================================================================
for CKPT_PATH in "${TRAINING_CHECKPOINTS[@]}"; do
    # Extract experiment name and step from checkpoint path
    # Expected format: .../checkpoints/<exp_name>/global_step_<N>
    EXP_DIR_NAME=$(basename "$(dirname "$CKPT_PATH")")
    STEP=$(basename "$CKPT_PATH" | sed 's/global_step_//')
    EXP_NAME="EVAL-${EXP_DIR_NAME}-step${STEP}"

    # Need to extract the original model path from the checkpoint config
    # For now, we'll use a default base model and let the checkpoint override weights
    # Alternatively, you can specify the base model path manually
    BASE_MODEL_PATH="/checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-27B"

    # Evaluation-only arguments with checkpoint resume
    ARGS="trainer.val_only=True \
trainer.val_before_train=True \
trainer.group_name=EVAL-rich-feedback \
trainer.resume_mode=resume_path \
trainer.resume_from_path=$CKPT_PATH \
actor_rollout_ref.model.path=$BASE_MODEL_PATH \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
trainer.nnodes=$NODES \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
algorithm.adv_estimator=grpo"

    submit_eval_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
done

echo ""
echo "================================================================"
echo "Evaluation jobs submitted. Check logs at:"
echo "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/"
echo "================================================================"
