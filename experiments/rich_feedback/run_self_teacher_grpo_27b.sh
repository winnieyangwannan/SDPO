#!/bin/bash

# Usage: ./run_self_teacher_grpo_27b.sh [--dry-run]

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base settings
CONFIG_NAME="self_teacher_grpo"
BASE_JOB_NAME="SELF_TEACHER_GRPO_27B"

ACCOUNT="agentic-models" # "aira_ws1" # "aira_ws2" # #
QOS="h200_agentic-models_high" # "h200_aira_ws1_high" # "h200_coding_shared" #

DATA_PATHS=(
    "lcb_v6"
)

# Fixed Slurm resources (2 nodes for 27B model)
NODES=2
TIME="168:00:00"
NTASKS_PER_NODE=1
GPUS_PER_NODE=8
MEM=460000
CPUS_PER_TASK=96

# Sweep Parameters
TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(32)
MINI_BATCH_SIZES=(16)

LRS=(1e-6)
SEEDS=(42 123 456)
SAVE_FREQ=50

# Directory to save raw validation generations (JSONL format)
VAL_DATA_DIR="/checkpoint/agentic-models/winnieyangwn/SDPO/$BASE_JOB_NAME/eval"

MODEL_PATHS=(
    "/checkpoint/agentic-models/winnieyangwn/models/Qwen3.5-27B"
)

# =============================================================================
# JOB SUBMISSION FUNCTION (Multi-node with Ray cluster)
# =============================================================================

submit_job() {
    local exp_name="$1"
    local script_args="$2"
    local data_path="$3"

    # For multi-node jobs, we need to set up a Ray cluster properly
    # Create a temporary script that handles Ray cluster initialization
    local job_script="/tmp/verl_multinode_${exp_name}.sh"
    
    cat > "$job_script" << 'SCRIPT_EOF'
#!/bin/bash
set -e

# Environment setup
eval "$(conda shell.bash hook)"
conda activate verl2
export PYTHONPATH=/home/$USER/SDPO:$PYTHONPATH
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DISABLE_METRICS=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

port=6379
ip_head=$head_node_ip:$port
export ip_head
export RAY_ADDRESS=$ip_head

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "IP Head: $ip_head"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Start Ray head on the first node
echo "Starting Ray HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "eval \"\$(conda shell.bash hook)\"; conda activate verl2; ray start --head --node-ip-address=\"$head_node_ip\" --port=$port --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block" &
sleep 20

# Start Ray workers on other nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash -c "eval \"\$(conda shell.bash hook)\"; conda activate verl2; ray start --address $ip_head --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus ${SLURM_GPUS_PER_NODE} --block" &
    sleep 10
done

# Wait for cluster to be ready
sleep 15
echo "Ray cluster started. Checking status..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" bash -c "eval \"\$(conda shell.bash hook)\"; conda activate verl2; ray status"

# Run training on head node
echo "Starting training..."
SCRIPT_EOF

    # Append the actual training command with variables expanded
    cat >> "$job_script" << SCRIPT_EOF
export BASE_JOB_NAME=$BASE_JOB_NAME
export RAY_ADDRESS=\$ip_head
export EXPERIMENT=$exp_name
srun --export=ALL --overlap --nodes=1 --ntasks=1 -w "\$head_node" \
    bash -c "eval \"\\\$(conda shell.bash hook)\"; conda activate verl2; export PYTHONPATH=/home/\$USER/SDPO:\\\$PYTHONPATH; export RAY_ADDRESS=\$ip_head; export EXPERIMENT=$exp_name; export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export RAY_DISABLE_METRICS=1; export VLLM_ATTENTION_BACKEND=XFORMERS; export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa; export BASE_JOB_NAME=$BASE_JOB_NAME; bash /home/\$USER/SDPO/training/verl_training.sh $exp_name $CONFIG_NAME $data_path $script_args"
SCRIPT_EOF

    chmod +x "$job_script"

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
        "$job_script"
    )

    if [ "$DRY_RUN" = true ]; then
        echo "----------------------------------------------------------------"
        echo "Would submit job for: $exp_name"
        echo "${sbatch_cmd[@]}"
    else
        # Ensure output directory exists
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs"
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/outputs"
        mkdir -p "/checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/checkpoints"

        echo "Submitting job for: $exp_name"
        job_output=$("${sbatch_cmd[@]}")
        echo "$job_output"
        job_id=$(echo "$job_output" | awk '{print $NF}')
        echo "  Log: /checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/${job_id}.log"
        echo "  Err: /checkpoint/agentic-models/$USER/SDPO/$BASE_JOB_NAME/logs/${job_id}.err"
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
                    for SEED in "${SEEDS[@]}"; do
                        for DATA_PATH in "${DATA_PATHS[@]}"; do
                            # 1. Construct the experiment name (must be unique)
                            MODEL_NAME="${MODEL_PATH##*/}"  # Extract name after last "/" (e.g., Qwen/Qwen3-8B -> Qwen3-8B)
                            EXP_NAME="SELF-TEACHER-GRPO-27B-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-seed${SEED}-model${MODEL_NAME}"

                            # 2. Construct the arguments string to pass to the training script
                            # Format: key=value key2=value2 ...
                            # Override vars that use env var interpolation to ensure they work in Ray
                            ARGS="vars.task=$DATA_PATH \
vars.dir=/checkpoint/agentic-models/winnieyangwn/SDPO/$BASE_JOB_NAME/outputs \
vars.log_dir=/checkpoint/agentic-models/winnieyangwn/SDPO/$BASE_JOB_NAME/logs \
vars.ckpt_dir=/checkpoint/agentic-models/winnieyangwn/SDPO/$BASE_JOB_NAME/checkpoints \
custom_reward_function.path=/home/winnieyangwn/SDPO/verl/utils/reward_score/feedback/__init__.py \
data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.n_gpus_per_node=$GPUS_PER_NODE \
trainer.group_name=SELF-TEACHER-GRPO-27B-rich-feedback \
trainer.experiment_name=$EXP_NAME \
trainer.test_freq=5 \
trainer.validation_save_freq=50 \
trainer.validation_data_dir=$VAL_DATA_DIR/$EXP_NAME \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.data_loader_seed=$SEED \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
actor_rollout_ref.rollout.val_kwargs.do_sample=True \
actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192 \
trainer.total_training_steps=300 \
trainer.nnodes=2 \
trainer.save_freq=$SAVE_FREQ \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
algorithm.self_teacher.enable=true \
algorithm.self_teacher.privilege_fraction=0.5 \
algorithm.self_teacher.privilege_fraction_decay=linear \
algorithm.self_teacher.accuracy_threshold=0.8 \
algorithm.self_teacher.privilege_penalty.enable=true \
algorithm.self_teacher.privilege_penalty.penalty_min=0.02 \
algorithm.self_teacher.privilege_penalty.penalty_max=0.10 \
algorithm.self_teacher.privilege_penalty.schedule=late_ramp \
algorithm.self_teacher.privilege_penalty.apply_to_grpo=true"

                            # 3. Submit
                            submit_job "$EXP_NAME" "$ARGS" "$DATA_PATH"
                        done
                    done
                done
            done
        done
    done
done

