#!/bin/bash
#SBATCH --job-name=sample_all_together_wa
#SBATCH --output=logs/sample_all_together_wa_%j.out
#SBATCH --error=logs/sample_all_together_wa_%j.err
#SBATCH --time=5:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

# Generate 150 samples per test sample for all_together model
echo "Generating 150 samples for all_together model (wa dataset)"
# python generate_samples_diffusion.py \
#   --model diffusion_xy \
#   --dataset wa \
#   --model_dataset wa \
#   --vf_conversion none \
#   --num_samples 150 \
#   --num_steps 52 \
#   --batch_size 256 \
#   --D 64 \
#   --K 2
python generate_samples_diffusion.py \
  --model diffusion_xy_var \
  --dataset wa \
  --model_dataset joint_orig \
  --max_cond 2 \
  --num_samples 150 \
  --num_steps 52 \
  --batch_size 256 \
  --L 59 \
  --D 64 \
  --K 2

echo "Sampling complete!"
