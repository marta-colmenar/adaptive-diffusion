#!/bin/bash
#SBATCH --job-name=eval_cross_attn_wa_4vf
#SBATCH --output=logs/eval_cross_attn_wa_4vf_%j.out
#SBATCH --error=logs/eval_cross_attn_wa_4vf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mail-user=marta.colmenar@unibe.ch
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:h100:1

echo "================================================"
echo "Evaluating cross_attn model on WA dataset (4VF)"
echo "Tasks: VF3→VF4, VF2,VF3→VF4, VF1,VF2,VF3→VF4"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /storage/homefs/mc21s017

source anaconda3/bin/activate
conda activate base

python evaluate_4vf_models.py \
    --model all_together \
    --dataset wa \
    --num_samples 1000 \
    --num_steps 52 \
    --batch_size 512 \
    --D 64 \
    --K 2 \
    --data_path /storage/homefs/mc21s017/data_vf/ \
    --model_path /storage/homefs/mc21s017/ \
    --output_dir /storage/homefs/mc21s017/results_4vf/

echo ""
echo "Job finished at: $(date)"
