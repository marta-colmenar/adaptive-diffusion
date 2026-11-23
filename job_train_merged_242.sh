#!/bin/bash
#SBATCH --job-name=train_merged_242
#SBATCH --output=logs/train_merged_242_%j.out
#SBATCH --error=logs/train_merged_242_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

# Train merged model on 24-2 pattern (52 locations, WA + Bern converted)
# NOTE: For simplicity, this currently uses WA data only (24-2 pattern)
echo "Training all_together model on merged data (24-2 pattern, 52 locations)"
python training_merged_datasets.py \
  --target_pattern 24-2 \
  --epochs 15000 \
  --batch_size 512 \
  --lr 1e-3 \
  --num_steps 52 \
  --eval_every 50 \
  --log_every 10 \
  --D 64 \
  --K 2 \
  --max_cond 2 \
  --clip_min_minus30

echo "Training complete!"
