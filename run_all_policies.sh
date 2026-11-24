#!/bin/bash
# run_all_policies.sh
# ====================
# Master launcher: submits the full pipeline for one dataset.
#
# Usage:
#   bash run_all_policies.sh bern 3
#   bash run_all_policies.sh wa   3
#
# Pipeline order:
#   1. Compute A6 population order  (job_pop_order.sh)
#   2. Sample all policies in parallel array jobs  (job_policy_array.sh)
#      -- pop_order waits for step 1 to finish
#   3. Merge parts  (merge_policy_parts.py, inline after each policy)
#   4. Compute metrics  (job_metrics.sh)
#   5. Plot  (inline, CPU-only)

DATASET=${1:-bern}
N_VF=${2:-3}
N_PARTS=4           # number of SLURM array tasks per policy
N_SAMPLES=2500      # total test samples (N_PARTS × SAMPLES_PER_JOB in job_policy_array.sh)
NUM_MC=100

RESULTS_DIR=/storage/homefs/mc21s017/results/
DATA_PATH=/storage/homefs/mc21s017/data_vf/
FIGS_DIR=${RESULTS_DIR}figures/

mkdir -p logs ${RESULTS_DIR} ${FIGS_DIR}

echo "========================================================"
echo " Adaptive Diffusion – Policy Comparison Pipeline"
echo " Dataset: ${DATASET}  n_vf: ${N_VF}"
echo "========================================================"

# ── Step 1: population order ──────────────────────────────────────────────────
echo ""
echo "[1/5] Submitting A6 population order job..."
POP_JOB=$(sbatch --parsable job_pop_order.sh ${DATASET} ${N_VF})
echo "      Job ID: ${POP_JOB}"

# ── Step 2: sampling (all policies in parallel) ───────────────────────────────
echo ""
echo "[2/5] Submitting sampling jobs..."

# Fixed policies (no dependency on pop order, batch inference)
for POLICY in random growing max_variance max_entropy mse_oracle; do
    JID=$(sbatch --parsable \
        --job-name="pol_${POLICY:0:8}" \
        job_policy_array.sh ${POLICY} ${DATASET} ${N_VF})
    echo "      ${POLICY}: Job ${JID}"
    eval "JID_${POLICY//_/}=${JID}"
done

# pop_order depends on pop order computation
JID_poporder=$(sbatch --parsable \
    --job-name="pol_pop_ord" \
    --dependency=afterok:${POP_JOB} \
    job_policy_array.sh pop_order ${DATASET} ${N_VF})
echo "      pop_order: Job ${JID_poporder} (after ${POP_JOB})"

# ── Step 3: merge parts (after all sampling done) ────────────────────────────
# Build dependency list
ALL_SAMPLE_JOBS="${JID_random}:${JID_growing}:${JID_maxvariance}:${JID_maxentropy}:${JID_mseoracle}:${JID_poporder}"

echo ""
echo "[3/5] Will merge parts after all sampling jobs complete."
echo "      Run manually when done:"
echo ""
for POLICY in random growing pop_order max_variance max_entropy mse_oracle; do
    echo "  python merge_policy_parts.py \\"
    echo "      --results_dir ${RESULTS_DIR} \\"
    echo "      --policy ${POLICY} --dataset ${DATASET} --n_vf ${N_VF} \\"
    echo "      --n_parts ${N_PARTS} --num_mc ${NUM_MC}"
done
echo ""

# ── Step 4: metrics ───────────────────────────────────────────────────────────
echo "[4/5] Submit metrics job (run after merging):"
echo "  sbatch job_metrics.sh ${DATASET} ${N_VF}"
echo ""

# ── Step 5: figures ───────────────────────────────────────────────────────────
echo "[5/5] Plot (run after metrics):"
echo "  python plot_policy_comparison.py \\"
echo "      --metrics_dir ${RESULTS_DIR}metrics/ \\"
echo "      --dataset ${DATASET} \\"
echo "      --data_path ${DATA_PATH} \\"
echo "      --output_dir ${FIGS_DIR} \\"
echo "      --budget_k 20"
echo ""
echo "========================================================"
echo " Sampling job IDs submitted. Monitor with:"
echo "   squeue -u \$USER"
echo "========================================================"
