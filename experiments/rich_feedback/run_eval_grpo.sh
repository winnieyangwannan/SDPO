#!/bin/bash

# =============================================================================
# Evaluation-Only Script for LiveCodeBench
# =============================================================================
# Usage: 
# 
#   ./experiments/rich_feedback/run_eval_only.sh --dry-run                    # Preview commands
#   ./experiments/rich_feedback/run_eval_only.sh                               # Submit jobs
# c hmod +x /home/winnieyangwn/SDPO/experiments/rich_feedback/run_eval_only.sh
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
BASE_JOB_NAME="GRPO"
SLURM_NAME="EVAL_GRPO"


ACCOUNT="agentic-models" # "aira_ws2" # #
QOS="h200_agentic-models_high" # "h200_coding_shared" # #  # "h200_aira_ws1_high" 

# Data path for evaluation
DATA_PATH="lcb_v6"

# Fixed Slurm resources (can use fewer resources for eval-only)
# NOTE: For eval-only, single node is sufficient as actor and rollout share GPUs
NODES=1
TIME="4:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Number of samples per prompt for evaluation
VAL_N=16

# Directory to save raw validation generations (JSONL format)
VAL_DATA_DIR="/checkpoint/agentic-models/winnieyangwn/SDPO/$BASE_JOB_NAME/eval"

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
    # "/checkpoint/agentic-models/winnieyangwn/SDPO/GRPO/checkpoints/FINAL-GRPO-mbs-16-train32-rollout8-lr1e-6-seed123-modelQwen3.5-27B/global_step_2"
    "/checkpoint/agentic-models/winnieyangwn/SDPO/GRPO/checkpoints/FINAL-GRPO-mbs-8-train32-rollout8-lr1e-6-seed123-modelQwen3.5-9B/global_step_50"
    # Add more checkpoint paths here
)

# =============================================================================
# CHECKPOINT CONVERSION FUNCTION
# =============================================================================
# Converts FSDP checkpoint to HuggingFace format if not already converted

convert_checkpoint_if_needed() {
    local ckpt_path="$1"
    local actor_path="${ckpt_path}/actor"
    local hf_path="${actor_path}/huggingface"
    
    # Check if model weights already exist
    if [[ -f "${hf_path}/model.safetensors" ]] || \
       [[ -f "${hf_path}/pytorch_model.bin" ]] || \
       ls "${hf_path}"/model-*.safetensors 1>/dev/null 2>&1; then
        echo "  [OK] HuggingFace weights already exist in ${hf_path}"
        return 0
    fi
    
    # Check if FSDP checkpoint exists
    if ! ls "${actor_path}"/model_world_size_*.pt 1>/dev/null 2>&1; then
        echo "  [ERROR] No FSDP checkpoint found in ${actor_path}"
        return 1
    fi
    
    echo "  [CONVERTING] FSDP checkpoint to HuggingFace format..."
    echo "    Source: ${actor_path}"
    echo "    Target: ${hf_path}"
    
    # Activate conda environment and run conversion
    (
        eval "$(conda shell.bash hook)"
        conda activate verl2
        export PYTHONPATH=/home/$USER/SDPO:$PYTHONPATH
        
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${actor_path}" \
            --target_dir "${hf_path}"
    )
    
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "  [OK] Checkpoint conversion completed successfully"
        return 0
    else
        echo "  [ERROR] Checkpoint conversion failed with exit code ${exit_code}"
        return 1
    fi
}

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_eval_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Define the environment setup and command execution
    local setup_cmds="eval \"\$(conda shell.bash hook)\"; conda activate verl2; export BASE_JOB_NAME=$BASE_JOB_NAME; export PYTHONPATH=/home/$USER/SDPO:\$PYTHONPATH; export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export RAY_DISABLE_METRICS=1; export VLLM_ATTENTION_BACKEND=XFORMERS; export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa; export WANDB_MODE=disabled"

    local run_cmd="bash /home/$USER/SDPO/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"

    local wrapped_cmd="srun bash -c '$setup_cmds; $run_cmd'"

    local sbatch_cmd=(
        sbatch
        --job-name="$SLURM_NAME"
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
        job_output=$("${sbatch_cmd[@]}")
        echo "$job_output"
        job_id=$(echo "$job_output" | awk '{print $NF}')
        echo "  Log: /checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/${job_id}.log"
        echo "  Err: /checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/${job_id}.err"
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
trainer.validation_data_dir=$VAL_DATA_DIR/$EXP_NAME \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
trainer.nnodes=$NODES \
trainer.n_gpus_per_node=$GPUS_PER_NODE \
actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
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

    echo "----------------------------------------------------------------"
    echo "Processing checkpoint: $CKPT_PATH"
    
    # Convert FSDP checkpoint to HuggingFace format if needed
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would check and convert checkpoint if needed"
    else
        if ! convert_checkpoint_if_needed "$CKPT_PATH"; then
            echo "  [SKIP] Skipping evaluation for $EXP_NAME due to conversion failure"
            continue
        fi
    fi

    # Use the checkpoint's huggingface directory as model path (has config.json and tokenizer)
    BASE_MODEL_PATH="${CKPT_PATH}/actor/huggingface"

    # Evaluation-only arguments - load directly from converted HF model
    # NOTE: We skip FSDP checkpoint loading (resume_from_path) since we already
    # converted to HuggingFace format. This avoids world_size mismatch issues.
    ARGS="trainer.val_only=True \
trainer.val_before_train=True \
trainer.group_name=EVAL-rich-feedback \
trainer.validation_data_dir=$VAL_DATA_DIR/$EXP_NAME \
actor_rollout_ref.model.path=$BASE_MODEL_PATH \
actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
trainer.nnodes=$NODES \
trainer.n_gpus_per_node=$GPUS_PER_NODE \
actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
algorithm.adv_estimator=grpo"

    submit_eval_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
done

echo ""
echo "================================================================"
echo "Evaluation jobs submitted. Check logs at:"
echo "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/"
echo "================================================================"
