#!/bin/bash
#SBATCH --job-name=sample_triplets_diffusion
#SBATCH --output=logs/sample_triplets_diffusion_%j.out
#SBATCH --error=logs/sample_triplets_diffusion_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
##SBATCH --gres=gpu:rtx4090:1
##SBATCH --qos=job_gpu_caim
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

set -euo pipefail

mkdir -p logs

echo "Generating triplet diffusion samples from triplets_oct_vf.parquet"
python generate_triplets_diffusion_samples.py 

echo "Sampling complete!"