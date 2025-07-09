#!/bin/bash



if [ -z "$1" ]; then
    echo "Usage: $0 <NUM_GPUS> <DATASET> [additional options]"
    echo "Example: $0 2 shapenet"
    echo "Example: $0 1 shapenet --opt training.epochs 500 data.batch_size 8"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Dataset argument required"
    echo "Usage: $0 <NUM_GPUS> <DATASET> [additional options]"
    exit 1
fi

# Parse arguments
NGPU=$1
DATASET=$2
shift 2  # Remove first two arguments, rest will be passed to train.py

# Configuration
num_node=1
BS=4
total_bs=$(( $NGPU * $BS ))

# Safety check for batch size
if (( $total_bs > 128 )); then 
    echo "[WARNING] Total batch_size ($total_bs) larger than 128 may lead to unstable training"
    echo "Consider reducing batch size or number of GPUs"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Base training command
BASE_CMD="python train.py --num_gpus $NGPU --dataset $DATASET"

# Default configuration overrides
DEFAULT_OPTS=(
    "--opt"
    "data.batch_size" "$BS"
    "data.num_workers" "4"
    "training.epochs" "500"
    "training.opt.lr" "1e-4"
    "training.opt.beta2" "0.99"
    "model.latent_dim" "256"
    "data.tr_max_sample_points" "3000"
    "data.te_max_sample_points" "3000"
    "data.random_subsample" "1"
    "data.recenter_per_shape" "False"
    "data.normalize_global" "True"
)

# Combine base command with default options and any additional arguments
FULL_CMD="$BASE_CMD ${DEFAULT_OPTS[*]} $@"

echo "========================================="
echo "DELIGHT Training Script"
echo "========================================="
echo "Number of GPUs: $NGPU"
echo "Dataset: $DATASET"
echo "Batch size per GPU: $BS"
echo "Total batch size: $total_bs"
echo "Number of nodes: $num_node"
echo "========================================="
echo "Running command:"
echo "$FULL_CMD"
echo "========================================="

# Execute the training command
eval $FULL_CMD