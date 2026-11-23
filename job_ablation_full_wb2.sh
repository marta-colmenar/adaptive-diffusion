#!/bin/bash
#SBATCH --job-name=octs_diffusion
#SBATCH --output=logs/octs_diffusion_%j.out
#SBATCH --error=logs/octs_diffusion_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
##SBATCH --qos=job_gpu_sznitman
##SBATCH --gres=gpu:h100:1
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --qos=job_gpu_caim

# Full model with learnable lambda
# echo "Training full model"
# python training_ablation_wa.py \
#     --model diffusion_xy_var \
#     --dataset joint_orig \
#     --epochs 15000 \
#     --batch_size 512 \
#     --lr 1e-3 \
#     --num_steps 52 \
#     --eval_every 50 \
#     --D 64 \
#     --K 2 \
#     --max_cond 2 \

echo "Training model octs"
python training_ablation_wa.py \
    --model diffusion_loc \
    --dataset oct_vf4 \
    --L 59 \
    --epochs 15000 \
    --batch_size 128 \
    --lr 1e-3 \
    --num_steps 52 \
    --eval_every 50 \
    --D 64 \
    --K 2 \
    --max_cond 2 \
    --train_task 2 \
    --no_wandb
