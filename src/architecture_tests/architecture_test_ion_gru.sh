#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="arch-test_ion_gru"
#SBATCH --output=architecture_test_ion_gru_output.txt

cd /home/malekia/idp-binding-site-prediction/data

source ~/miniconda3/etc/profile.d/conda.sh
conda activate protein

python3 /home/malekia/idp-binding-site-prediction/src/architecture_test_ion_gru.py