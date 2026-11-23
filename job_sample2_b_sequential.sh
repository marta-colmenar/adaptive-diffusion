#!/bin/bash
#SBATCH --job-name=greedy_array
#SBATCH --output=logs/greedy_%A_%a.out
#SBATCH --error=logs/greedy_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_caim
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --array=0-3

# Generate 150 samples per test sample for all_together model
echo "Generating 150 samples for all_together model (wa dataset)"

module load Anaconda3
eval "$(conda shell.bash hook)"

SAMPLES_PER_JOB=625
OFFSET=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_JOB ))

python sample_sequential_ordered_prefix_mc.py \
  --dataset bern \
  --n_vf 3 \
  --checkpoint /storage/homefs/mc21s017/sequential_observed_bern_3vf.pth \
  --num_mc 100 \
  --batch_size 64 \
  --mc_chunk 100 \
  --sample_offset $OFFSET \
  --max_test_samples $SAMPLES_PER_JOB \
  --output_name "difussion_greedy_part_${SLURM_ARRAY_TASK_ID}" \
  --greedy

echo "Sampling complete!"
