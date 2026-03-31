#!/bin/bash
# =============================================================================
# COMMON UTILITIES FOR JOB SUBMISSION SCRIPTS
# =============================================================================
# Source this file from your run_*.sh scripts:
#   source "$(dirname "$0")/common.sh"

# =============================================================================
# DRY RUN PARSING
# =============================================================================

parse_dry_run() {
    DRY_RUN=false
    if [[ "$1" == "--dry-run" ]]; then
        DRY_RUN=true
        echo "Dry run mode enabled. Commands will be printed but not executed."
    fi
}

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # Define the environment setup and command execution
    # We use the user's home directory dynamically
    local base_job_export=""
    if [ -n "$BASE_JOB_NAME" ]; then
        base_job_export="export BASE_JOB_NAME=$BASE_JOB_NAME; "
    fi

    local setup_cmds="eval \"\$(conda shell.bash hook)\"; conda activate sdpo2; ${base_job_export}export PYTHONPATH=/home/$USER/SDPO:\$PYTHONPATH; export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export RAY_DISABLE_METRICS=1; export VLLM_ATTENTION_BACKEND=XFORMERS; export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa"
    # local setup_cmds="eval \"\$(conda shell.bash hook)\"; conda activate verl2; ${base_job_export}export PYTHONPATH=/home/$USER/SDPO:\$PYTHONPATH; export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export RAY_DISABLE_METRICS=1; export VLLM_ATTENTION_BACKEND=XFORMERS; export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa"

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
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        # Ensure output directory exists
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/outputs"
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/checkpoints"
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs"

        echo "Submitting job for: $exp_name"
        "${sbatch_cmd[@]}"
    fi
}
