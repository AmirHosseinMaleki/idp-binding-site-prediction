#!/bin/bash
#SBATCH --account=ksibio
#SBATCH --partition=gpu-bio
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="embeddings"
#SBATCH --output=embeddings_output.txt

cd /home/malekia/idp-binding-site-prediction/data

# source /home/malekia/idp-binding-site-prediction/biotite_venv/bin/activate
source ~/miniconda3/etc/profile.d/conda.sh

conda activate protein

which python
python --version

python3 /home/malekia/idp-binding-site-prediction/src/generate_embeddings.py

# python3 /home/malekia/idp-binding-site-prediction/src/4_train.py

# python3 /home/malekia/idp-binding-site-prediction/src/5_evaluate.py

# srun --pty -p gpu-ffa -c1 --gpus=1 --mem=512G bash
