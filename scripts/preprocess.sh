#!/bin/bash

# SLURM Options
#SBATCH -o delight.out-%j
#SBATCH -c 16

source /etc/profile
module load anaconda/2022b

DATA_DIR="../ShapeNetCore.v2"
SAVE_DIR="./data"
CATEGORIES=("all")  # Change to specific categories if needed

python datasets/preprocessing.py \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --dataset ShapeNetCore.v2 \
    --categories ${CATEGORIES[@]} \
    --n_processes 16 \
    --train_split 0.8 \
    --test_split 0.1