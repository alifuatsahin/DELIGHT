#!/bin/bash

# SLURM Options
#SBATCH -o delight.out-%j
#SBATCH -c 10
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/Python-ML-2025a
module load cuda/12.2
module load nccl/2.23.4-cuda12.2

CKPT="${CKPT:-}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"

if [ -z "$CKPT" ]; then
    echo "ERROR: set CKPT to a trained prior checkpoint, e.g."
    echo "  CKPT=../experiments/prior/<exp>/checkpoints/best_eval.pth bash scripts/test.sh"
    exit 1
fi

python main.py \
    --pretrained "$CKPT" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size 16 \
    --render \
    --metrics \
    "$@"
