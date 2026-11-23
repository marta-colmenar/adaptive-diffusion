#!/bin/bash
#SBATCH --job-name=create_triplets
#SBATCH --output=logs/create_triplets_%j.out
#SBATCH --error=logs/create_triplets_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=epyc2

# Create triplets from visual field data
echo "Creating triplet combinations from visual field data..."
cd /storage/homefs/mc21s017/data_vf
python create_triplets.py

echo "Triplet generation complete!"
