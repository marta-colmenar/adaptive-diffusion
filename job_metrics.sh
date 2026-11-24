#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --output=logs/metrics_%j.out
#SBATCH --error=logs/metrics_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=cpu

# Compute metrics for all policies.
# Run after all sampling jobs have completed and parts have been merged.
# Expects one merged _pred.npy / _target.npy / _timing.npy per policy.

DATASET=${1:-bern}
N_VF=${2:-3}
DATA_PATH=/storage/homefs/mc21s017/data_vf/
RESULTS_DIR=/storage/homefs/mc21s017/results/
METRICS_DIR=/storage/homefs/mc21s017/results/metrics/

module load Anaconda3
eval "$(conda shell.bash hook)"

mkdir -p ${METRICS_DIR}

POLICIES=(random growing pop_order max_variance max_entropy mse_oracle)

for POLICY in "${POLICIES[@]}"; do
    META=${RESULTS_DIR}policy_${POLICY}_${DATASET}_${N_VF}vf_offset0_mc100_meta.npz
    if [[ ! -f "${META}" ]]; then
        echo "[SKIP] No meta file for policy=${POLICY}: ${META}"
        continue
    fi
    echo "Computing metrics for policy=${POLICY} ..."
    python compute_metrics_policies.py \
        --meta_file  ${META} \
        --dataset    ${DATASET} \
        --n_vf       ${N_VF} \
        --data_path  ${DATA_PATH} \
        --output_dir ${METRICS_DIR}
done

echo "All metrics done!"
