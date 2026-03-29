#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="embed-dna-rna"
#SBATCH --output=embeddings_dna_rna_output.txt

cd /home/malekia/idp-binding-site-prediction/data

# Initialize and activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein

# Run embedding generation
python3 /home/malekia/idp-binding-site-prediction/src/generate_embeddings_dna_rna.py