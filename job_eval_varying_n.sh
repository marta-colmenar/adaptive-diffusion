#!/bin/bash
#SBATCH --job-name=eval_varying_n_bern
#SBATCH --output=logs/eval_varying_n_%j.out
#SBATCH --error=logs/eval_varying_n_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

# Load required modules
module load Anaconda3
source activate

# Run evaluation
python evaluate_varying_n_models.py --dataset bern

echo "Evaluation completed!"
