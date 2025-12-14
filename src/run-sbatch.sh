#!/bin/bash
#SBATCH --partition=gpu-ffa
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=512000
#SBATCH --gpus=1                  
#SBATCH --job-name="ion-training"
#SBATCH --output=training_output.txt

cd /home/malekia/idp-binding-site-prediction/data

source /home/malekia/idp-binding-site-prediction/biotite_venv/bin/activate

python3 /home/malekia/idp-binding-site-prediction/src/evaluate.py

# python3 /home/malekia/idp-binding-site-prediction/src/4_train.py

# python3 /home/malekia/idp-binding-site-prediction/src/5_evaluate.py

# srun --pty -p gpu-ffa -c1 --gpus=1 --mem=16G bash
