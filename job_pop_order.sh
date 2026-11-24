#!/bin/bash
#SBATCH --job-name=pop_order
#SBATCH --output=logs/pop_order_%j.out
#SBATCH --error=logs/pop_order_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_caim
#SBATCH --gres=gpu:rtx4090:1

# Compute A6 population-optimised order from training set.
# Must complete BEFORE the pop_order sampling jobs.

DATASET=${1:-bern}
N_VF=${2:-3}
CHECKPOINT=/storage/homefs/mc21s017/sequential_observed_${DATASET}_${N_VF}vf.pth
OUTPUT_DIR=/storage/homefs/mc21s017/results/

echo "Computing population order for dataset=${DATASET}, n_vf=${N_VF}"

module load Anaconda3
eval "$(conda shell.bash hook)"

mkdir -p logs
mkdir -p ${OUTPUT_DIR}

python compute_population_order.py \
    --dataset      ${DATASET} \
    --n_vf         ${N_VF} \
    --checkpoint   ${CHECKPOINT} \
    --num_mc       50 \
    --mc_chunk     50 \
    --batch_size   64 \
    --num_workers  4 \
    --output_dir   ${OUTPUT_DIR}

echo "Population order computation complete!"
