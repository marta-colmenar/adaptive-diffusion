#!/bin/bash
#SBATCH --job-name=policy_array
#SBATCH --output=logs/policy_%x_%A_%a.out
#SBATCH --error=logs/policy_%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_caim
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --array=0-3

# SLURM array job for a single policy.
# 4 array tasks × 625 samples = 2500 patients (adjust SAMPLES_PER_JOB as needed).
#
# Submit with:
#   sbatch --job-name=policy_random       job_policy_array.sh random    bern 3
#   sbatch --job-name=policy_growing      job_policy_array.sh growing   bern 3
#   sbatch --job-name=policy_pop_order    job_policy_array.sh pop_order bern 3
#   sbatch --job-name=policy_max_variance job_policy_array.sh max_variance bern 3
#   sbatch --job-name=policy_max_entropy  job_policy_array.sh max_entropy  bern 3
#   sbatch --job-name=policy_mse_oracle   job_policy_array.sh mse_oracle   bern 3

POLICY=${1:-random}
DATASET=${2:-bern}
N_VF=${3:-3}

CHECKPOINT=/storage/homefs/mc21s017/sequential_observed_${DATASET}_${N_VF}vf.pth
DATA_PATH=/storage/homefs/mc21s017/data_vf/
OUTPUT_DIR=/storage/homefs/mc21s017/results/
POP_ORDER_FILE=${OUTPUT_DIR}pop_order_${DATASET}_${N_VF}vf.npy

SAMPLES_PER_JOB=625
OFFSET=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_JOB ))

# For fixed (non-greedy) policies we can afford larger batch sizes
if [[ "${POLICY}" == "growing" || "${POLICY}" == "pop_order" ]]; then
    BATCH_SIZE=64
else
    BATCH_SIZE=1
fi

echo "Policy=${POLICY}  Dataset=${DATASET}  n_vf=${N_VF}"
echo "Array task ${SLURM_ARRAY_TASK_ID}: offset=${OFFSET}, samples=${SAMPLES_PER_JOB}"

module load Anaconda3
eval "$(conda shell.bash hook)"

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

# Build optional --order_file argument for pop_order policy
ORDER_FLAG=""
if [[ "${POLICY}" == "pop_order" ]]; then
    if [[ ! -f "${POP_ORDER_FILE}" ]]; then
        echo "ERROR: pop_order file not found: ${POP_ORDER_FILE}"
        echo "Run job_pop_order.sh first."
        exit 1
    fi
    ORDER_FLAG="--order_file ${POP_ORDER_FILE}"
fi

python sample_policies_mc.py \
    --dataset           ${DATASET} \
    --n_vf              ${N_VF} \
    --checkpoint        ${CHECKPOINT} \
    --policy            ${POLICY} \
    ${ORDER_FLAG} \
    --num_mc            100 \
    --mc_chunk          100 \
    --batch_size        ${BATCH_SIZE} \
    --num_workers       4 \
    --sample_offset     ${OFFSET} \
    --max_test_samples  ${SAMPLES_PER_JOB} \
    --data_path         ${DATA_PATH} \
    --output_dir        ${OUTPUT_DIR} \
    --output_name       "policy_${POLICY}_${DATASET}_${N_VF}vf_part${SLURM_ARRAY_TASK_ID}"

echo "Sampling complete for array task ${SLURM_ARRAY_TASK_ID}!"
