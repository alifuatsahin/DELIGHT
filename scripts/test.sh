#!/bin/bash

# Example script for testing/running main.py
# Update paths as needed for your setup

# SLURM Options (uncomment if using SLURM)
#SBATCH -o delight.out-%j
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

# Load modules (uncomment if using module system)
# source /etc/profile
# module load anaconda/Python-ML-2025a
# module load cuda/12.2
# module load nccl/2.23.4-cuda12.2

# Example usage:
# python3 main.py --config configs/example_vae.yml --input_pc path/to/input.npy

python3 main.py --help