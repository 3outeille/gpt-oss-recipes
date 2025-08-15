#!/bin/bash
#SBATCH --job-name=bench-gpt-oss-20b
#SBATCH --time=00:30:00
#SBATCH --partition=hopper-prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# Example of running the job:
# sbatch template.slurm configs/tp_lora_megablocks_flash.yaml

source ../gpt-oss/bin/activate
source /etc/profile.d/modules.sh
module load cuda/12.9

# Distributed configuration
NUM_NODES=1
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: sbatch $0 <config_file>"
  exit 1
fi

# Derive log file name from config file
LOG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
JOB_ID=${SLURM_JOB_ID}
LOG_FILE="${LOG_BASENAME}_${JOB_ID}.out"

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

echo "Running command: $CMD"
srun -u $CMD &> "$LOG_FILE"

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"