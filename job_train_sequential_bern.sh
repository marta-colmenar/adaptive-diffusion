#!/bin/bash
#SBATCH --job-name=train_seq_obs_bern
#SBATCH --output=logs/train_seq_obs_bern_%j.out
#SBATCH --error=logs/train_seq_obs_bern_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
##SBATCH --qos=job_gpu_caim
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=job_gpu_sznitman
##SBATCH --gres=gpu:rtx4090:1

echo "=============================================="
echo "Training sequential observed-location model"
echo "Dataset: bern"
echo "Testing via rollout sampling after training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /storage/homefs/mc21s017

module load Anaconda3
eval "$(conda shell.bash hook)"

python train_sequential_observed.py \
  --dataset bern \
  --epochs 10000 \
  --batch_size 512 \
  --lr 1e-3 \
  --max_cond 2 \
  --D 64 \
  --layers 2 \
  --sample_max_samples 200 \
  --data_path /storage/homefs/mc21s017/data_vf/ \
  --output_dir /storage/homefs/mc21s017/ \
  --init_checkpoint denoiser_ablation_diffusion_loc_vf_small_3vf.pth 

echo ""
echo "Job finished at: $(date)"