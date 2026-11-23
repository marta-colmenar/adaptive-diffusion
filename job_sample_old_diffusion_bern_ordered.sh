#!/bin/bash
#SBATCH --job-name=sample_old_diffusion_bern_ordered
#SBATCH --output=logs/sample_old_diffusion_bern_ordered_%j.out
#SBATCH --error=logs/sample_old_diffusion_bern_ordered_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

set -euo pipefail

echo "Generating ordered-prefix MC samples for the old Bern diffusion checkpoint"

python sample_sequential_ordered_prefix_mc.py \
  --dataset bern \
  --n_vf 3 \
  --checkpoint /storage/homefs/mc21s017/denoiser_ablation_diffusion_loc_bern_3vf.pth \
  --output_name sequential_prefix_mc_bern_3vf_old_diffusion_ordered_loc_novf3 \
  --num_mc 100 \
  --batch_size 256 \
  --mc_chunk 10

echo "Sampling complete!"