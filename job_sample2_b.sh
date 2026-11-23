#!/bin/bash
#SBATCH --job-name=sample_all_togethere_bern
#SBATCH --output=logs/sample_all_together_bern_%j.out
#SBATCH --error=logs/sample_all_together_bern_%j.err
#SBATCH --time=7:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
##SBATCH --partition=epyc2
#SBATCH --partition=gpu-invest
##SBATCH --gres=gpu:rtx4090:1
##SBATCH --qos=job_gpu_caim
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

# Generate 150 samples per test sample for all_togethere model
echo "Generating 150 samples for all_togethere model (bern dataset)"
python generate_samples_diffusion.py \
  --model diffusion_loc \
  --dataset bern \
  --model_dataset bern \
  --vf_conversion none \
  --num_samples 2000 \
  --num_steps 52 \
  --batch_size 256 \
  --D 64 \
  --K 2 \
  --L 59

# python generate_samples_diffusion.py \
#   --model diffusion_xy_var \
#   --dataset bern \
#   --model_dataset joint_orig \
#   --max_cond 2 \
#   --num_samples 2000 \
#   --num_steps 52 \
#   --batch_size 256 \
#   --L 59 \
#   --D 64 \
#   --K 2

echo "Sampling complete!"
