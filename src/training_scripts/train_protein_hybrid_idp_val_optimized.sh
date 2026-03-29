#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="protein-hybrid-idp-val_optimized"
#SBATCH --output=train_protein_hybrid_idp_val_optimized_output.txt

cd /home/malekia/idp-binding-site-prediction/data

source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein

python3 /home/malekia/idp-binding-site-prediction/src/train_protein_hybrid_idp_val_optimized.py