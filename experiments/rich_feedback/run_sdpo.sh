#!/bin/bash

# Usage: ./run_sdpo.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="sdpo"
BASE_JOB_NAME="rlvr"

ACCOUNT="agentic-models"
QOS="h200_agentic-models_high"

DATA_PATHS=(
    "lcb_v6"
)

# Fixed Slurm resources
NODES=1
TIME="168:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(1)
LRS=(1e-6)

# SDPO-specific parameters
# 0: forward KL, 0.5: Jensen-Shannon divergence, 1: reverse KL
ALPHAS=(1.0)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
SEEDS=(42 123 456)

MODEL_PATHS=(
    "/checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-9B"
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
        --output="/checkpoint/agentic-models/$USER/output/SDPO/%j.log"
        --error="/checkpoint/agentic-models/$USER/output/SDPO/%j.err"
        --wrap="$wrapped_cmd"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        # Ensure output directory exists
        mkdir -p "/checkpoint/agentic-models/$USER/output/SDPO"
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
                    for ALPHA in "${ALPHAS[@]}"; do
                        for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                            for SEED in "${SEEDS[@]}"; do
                                for DATA_PATH in "${DATA_PATHS[@]}"; do
                                    # 1. Construct the experiment name (must be unique)
                                    MODEL_NAME="${MODEL_PATH##*/}"  # Extract name after last "/" (e.g., Qwen/Qwen3-8B -> Qwen3-8B)
                                    EXP_NAME="FINAL-SDPO-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-seed${SEED}-model${MODEL_NAME}"

                                    # 2. Construct the arguments string to pass to the training script
                                    # Format: key=value key2=value2 ...
                                    ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SDPO-rich-feedback \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.data_loader_seed=$SEED \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
trainer.total_training_steps=300 \
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

