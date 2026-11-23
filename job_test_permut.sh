#!/bin/bash
#SBATCH --job-name=test_permut
#SBATCH --output=logs/test_permut_%j.out
#SBATCH --error=logs/test_permut_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
##SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
##SBATCH --gres=gpu:rtx4090:1
##SBATCH --qos=job_gpu_caim
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

python test_permutation.py