#!/bin/bash
#SBATCH --job-name=ablation_full_bern_vf3_loc
#SBATCH --output=logs/ablation_all_together_vf4_wa_%j.out
#SBATCH --error=logs/ablation_all_together_vf4_wa_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1
##SBATCH --gres=gpu:rtx4090:1
##SBATCH --qos=job_gpu_caim

# Full model with learnable lambda
echo "ablation_full_bern_vf3_loc"
python training_ablation_wa.py \
    --model diffusion_loc \
    --dataset vf_small \
    --L 59 \
    --epochs 15000 \
    --batch_size 128 \
    --lr 1e-3 \
    --num_steps 52 \
    --eval_every 25 \
    --D 64 \
    --K 2 \
    --max_cond 2 \
    --train_task 2 \
    --no_wandb 
