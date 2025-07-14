#!/bin/bash

# SLURM Options
#SBATCH -o delight.out-%j
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1
#SBATCH -p debug-gpu

export PYTHONPATH="~/Pasta/DELIGHT:~/DELIGHT/latent_diffusion:$PYTHONPATH"

source /etc/profile
module load anaconda/Python-ML-2025a
module load cuda/12.2
module load nccl/2.23.4-cuda12.2

python3 main.py