#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="protein-grid"
#SBATCH --output=protein_grid_search_output.txt

cd /home/malekia/idp-binding-site-prediction/data

source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein

python3 /home/malekia/idp-binding-site-prediction/src/grid_search_protein.py