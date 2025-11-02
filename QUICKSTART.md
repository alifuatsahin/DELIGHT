# DELIGHT Quick Start Guide

This guide will help you get up and running with DELIGHT quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (8GB+ VRAM recommended)
- CUDA 11.8 or higher
- 20GB+ free disk space

## Installation (5 minutes)

```bash
# Clone the repository
git clone https://github.com/alifuatsahin/DELIGHT.git
cd DELIGHT

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Compile CUDA extensions
cd third_party/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../../PyTorchEMD
python setup.py install
cd ../..
```

## Quick Dataset Setup

For a quick start, you can use a small subset of ShapeNet:

```bash
# Download a small sample (if available)
# Or prepare your own point cloud data in .npy format
mkdir -p ShapeNet/03001627/train
# Place your .npy files (shape: N x 3) in the train directory
```

## Train Your First Model (30 minutes)

### Option 1: Train VAE with Default Settings

```bash
python train.py \
    --num_gpus 1 \
    --opt \
    data.categories chair \
    data.batch_size 16 \
    training.epochs 100 \
    training.type vae
```

### Option 2: Use the Training Script

```bash
# Edit scripts/train_vae.sh to adjust settings
bash scripts/train_vae.sh
```

## Monitor Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir experiments/
```

Open http://localhost:6006 in your browser to view:
- Loss curves
- Generated samples
- Reconstruction quality

## Generate Your First Samples

After training completes, generate samples:

```bash
python train.py \
    --eval \
    --pretrained experiments/vae_softvq/your_experiment/checkpoints/best_eval.pth \
    --ntest 50
```

Results will be saved in the experiment directory.

## Common Issues and Solutions

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
--opt data.batch_size 8
```

### Issue: ChamferDistance compilation fails
**Solution**: Ensure CUDA_HOME is set
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Issue: Data loading is slow
**Solution**: Reduce number of workers
```bash
--opt data.num_workers 4
```

## Next Steps

1. **Train a diffusion prior**: See full README for DDPM training
2. **Experiment with configurations**: Modify `configs/example_config.yaml`
3. **Try different categories**: chair, airplane, car, etc.
4. **Adjust model size**: Modify `vae.latent_dim`, `vae.depth`, etc.

## Configuration Quick Reference

Key parameters you might want to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.batch_size` | 32 | Batch size (reduce if OOM) |
| `data.n_sample_points` | 2048 | Points per shape |
| `training.epochs` | 500 | Total training epochs |
| `training.opt.lr` | 0.0001 | Learning rate |
| `vae.latent_dim` | 512 | Latent space dimension |
| `vae.quantizer` | softvq | Quantization method |

## Getting Help

- 📖 Read the [full README](README.md)
- 🐛 [Report bugs](https://github.com/alifuatsahin/DELIGHT/issues)
- 💬 [Ask questions](https://github.com/alifuatsahin/DELIGHT/issues/new?template=question.md)
- 📚 Check [Contributing Guide](CONTRIBUTING.md)

## Example Commands Cheat Sheet

```bash
# Train VAE on chairs (single GPU)
python train.py --num_gpus 1 --opt data.categories chair training.type vae

# Train VAE on multiple categories (multi-GPU)
python train.py --num_gpus 2 --opt data.categories "chair,table" training.type vae

# Resume training from checkpoint
python train.py --resume --pretrained path/to/checkpoint.pth

# Evaluate model
python train.py --eval --pretrained path/to/checkpoint.pth --ntest 100

# Train with custom config
python train.py --config configs/example_config.yaml

# Override config from command line
python train.py --config configs/example_config.yaml --opt vae.latent_dim 256
```

Happy generating! 🎨✨
