#!/bin/bash

# SLURM Options
#SBATCH -o delight.out-%j
#SBATCH -c 10
#SBATCH --gres=gpu:volta:2

source /etc/profile
module load anaconda/Python-ML-2025a
module load cuda/12.2
module load nccl/2.23.4-cuda12.2

export TORCH_CUDA_ARCH_LIST="7.0"
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CKPT="../experiments/vae_softvq/airplane_bs32_20250825_033802/checkpoints/best_eval.pth"
NGPU=2 # 
num_node=1
BS=32
total_bs=$(( $NGPU * $BS ))

# Base training command
BASE_CMD="python train.py --num_gpus $NGPU --vae_checkpoint $CKPT"

# Default configuration overrides
DEFAULT_OPTS=(
    "--opt"
    "data.batch_size" "$BS"
    "data.num_workers" "10"
    "training.epochs" "600"
    "training.opt.lr" "1e-4"
    "training.type" "prior"
    "prior.depth" "4"
    "prior.width" "512"  # Feature dimension for the flow model
    "vis.vis_freq" "-50"
    "vis.save_freq" "50"
    "vis.val_freq" "25"
    "vis.save_time" "30"  # minutes
)

# Combine base command with default options and any additional arguments
FULL_CMD="$BASE_CMD ${DEFAULT_OPTS[*]} $@"

echo "========================================="
echo "DELIGHT Training Script"
echo "========================================="
echo "Number of GPUs: $NGPU"
echo "VAE: $CKPT"
echo "Batch size per GPU: $BS"
echo "Total batch size: $total_bs"
echo "Number of nodes: $num_node"
echo "========================================="
echo "Running command:"
echo "$FULL_CMD"
echo "========================================="

# Execute the training command
eval $FULL_CMD