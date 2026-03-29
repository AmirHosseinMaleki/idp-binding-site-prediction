#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="train-phase3"
#SBATCH --output=training_phase3_output.txt

cd /home/malekia/idp-binding-site-prediction/data

# Initialize and activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein

# Run training
python3 /home/malekia/idp-binding-site-prediction/src/train_phase3_esm.py