#!/bin/bash
#SBATCH --job-name=sample_merged_allxall
#SBATCH --output=logs/sample_merged_allxall_%j.out
#SBATCH --error=logs/sample_merged_allxall_%j.err
#SBATCH --time=5:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

set -euo pipefail

echo "Sampling merged_G on WA test (24toG)"
python generate_samples_diffusion.py \
  --model all_together \
  --dataset wa \
  --model_dataset merged_G_clipm30 \
  --vf_conversion auto \
  --num_samples 150 \
  --num_steps 52 \
  --batch_size 256 \
  --D 64 \
  --K 2

echo "Sampling merged_G on Bern test (none)"
python generate_samples_diffusion.py \
  --model all_together \
  --dataset bern \
  --model_dataset merged_G_clipm30 \
  --vf_conversion auto \
  --num_samples 150 \
  --num_steps 52 \
  --batch_size 256 \
  --D 64 \
  --K 2

echo "Sampling merged_24-2 on WA test (none)"
python generate_samples_diffusion.py \
  --model all_together \
  --dataset wa \
  --model_dataset merged_24-2_clipm30 \
  --vf_conversion auto \
  --num_samples 150 \
  --num_steps 52 \
  --batch_size 256 \
  --D 64 \
  --K 2

echo "Sampling merged_24-2 on Bern test (Gto24)"
python generate_samples_diffusion.py \
  --model all_together \
  --dataset bern \
  --model_dataset merged_24-2_clipm30 \
  --vf_conversion auto \
  --num_samples 150 \
  --num_steps 52 \
  --batch_size 256 \
  --D 64 \
  --K 2

echo "All four sampling runs completed."