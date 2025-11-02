# Frequently Asked Questions (FAQ)

## General Questions

### What is DELIGHT?

DELIGHT (Deep ComprEssion Latent dIffusion for Generation of High-quality Three-dimensional shapes) is a deep learning framework for generating 3D point clouds. It combines VQ-VAE for efficient compression and Diffusion Models for high-quality generation.

### What can I use DELIGHT for?

- 3D shape generation from scratch
- Point cloud reconstruction and compression
- Shape interpolation and manipulation
- Research in 3D generative models

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 8GB GPU memory
- 20GB disk space
- CUDA 11.8+

**Recommended:**
- Python 3.10+
- 16GB+ GPU memory
- 100GB+ disk space (for full ShapeNet)
- CUDA 12.0+

## Installation Issues

### Q: ChamferDistance compilation fails

**A:** Ensure CUDA is properly installed and environment variables are set:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA
nvcc --version

# Try compilation again
cd third_party/ChamferDistancePytorch/chamfer3D
python setup.py install
```

### Q: PyTorchEMD compilation fails

**A:** EMD requires a compatible C++ compiler:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Check compiler
g++ --version

# Ensure PyTorch is installed first
pip install torch torchvision

# Try compilation
cd third_party/PyTorchEMD
python setup.py install
```

### Q: ImportError: cannot import name 'xxx'

**A:** Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

If the error persists, try installing missing packages individually.

## Training Issues

### Q: CUDA out of memory error

**A:** Several solutions:

1. **Reduce batch size:**
   ```bash
   --opt data.batch_size 16  # or 8
   ```

2. **Reduce number of points:**
   ```bash
   --opt data.n_sample_points 1024
   ```

3. **Enable gradient checkpointing:**
   ```bash
   --opt ddpm.use_checkpoint True
   ```

4. **Use a smaller model:**
   ```bash
   --opt vae.latent_dim 256 vae.depth 10
   ```

### Q: Training is very slow

**A:** Several optimization strategies:

1. **Increase batch size** (if memory allows)
2. **Reduce number of workers** if CPU is bottleneck:
   ```bash
   --opt data.num_workers 4
   ```
3. **Use mixed precision training** (requires code modification)
4. **Enable xformers attention:**
   ```bash
   --opt ddpm.use_xformers_attention True
   ```

### Q: Loss is NaN or exploding

**A:** Try these solutions:

1. **Reduce learning rate:**
   ```bash
   --opt training.opt.lr 5e-5
   ```

2. **Enable gradient clipping** (add to trainer code)
3. **Check data normalization** settings
4. **Use KL annealing** for VAE:
   ```bash
   --opt vae.anneal_kl True
   ```

### Q: Model doesn't improve after many epochs

**A:** Potential causes:

1. **Learning rate too low** - increase it
2. **Model too small** - increase capacity
3. **Data issues** - check data loading and preprocessing
4. **Wrong loss weights** - adjust quantization loss weight

### Q: How do I resume training?

**A:**

```bash
python train.py \
    --resume \
    --pretrained path/to/checkpoint.pth
```

Or if the checkpoint is in the same experiment directory, it will auto-resume.

## Data Issues

### Q: Where can I get the ShapeNet dataset?

**A:** Download from the official website: https://shapenet.org/

You'll need to:
1. Register for an account
2. Download ShapeNet Core v2
3. Process into point cloud format (.npy files)

### Q: Can I use my own dataset?

**A:** Yes! You need to:

1. Convert your data to .npy format (N x 3 arrays)
2. Organize following the ShapeNet structure:
   ```
   data/
   ├── category_id/
   │   ├── train/
   │   │   ├── model1.npy
   │   │   └── ...
   │   ├── val/
   │   └── test/
   ```
3. Update `datasets/data_path.py` to point to your data
4. Adjust normalization settings in config

### Q: Data loading is very slow

**A:**

1. **Reduce num_workers** (paradoxically, too many can be slow):
   ```bash
   --opt data.num_workers 4
   ```

2. **Preload all data** into memory (if it fits)
3. **Use SSD** instead of HDD for data storage
4. **Check OVERFIT flag** - set to 0 for full dataset

### Q: What data preprocessing is recommended?

**A:** Standard preprocessing includes:

- **Centering**: `recenter_per_shape = True`
- **Normalization**: Choose one of:
  - `normalize_per_shape = True` (per-shape normalization)
  - `normalize_global = True` (dataset-wide normalization)
- **Sampling**: `n_sample_points = 2048` (adjust based on detail needed)

## Model Usage

### Q: How do I generate samples?

**A:**

```python
from models import VAE
import torch

# Load model
model = VAE(cfg)
model.load_state_dict(torch.load('checkpoint.pth')['model'])
model.eval()

# Generate samples
samples, labels = model.sample(n_sampled_points=2048, n_samples=10)
```

### Q: How do I reconstruct a point cloud?

**A:**

```python
# Load your point cloud
pc = torch.load('point_cloud.pt')  # Shape: (B, N, 3)

# Reconstruct
with torch.no_grad():
    recon, labels = model.recont(pc)
```

### Q: Can I interpolate between shapes?

**A:** Yes! Use the interpolate method:

```python
pc1 = load_point_cloud('shape1.npy')
pc2 = load_point_cloud('shape2.npy')

interpolated = model.interpolate(pc1, pc2, n_steps=10)
```

### Q: What's the difference between SoftVQ and KL quantizers?

**A:**

**SoftVQ:**
- Soft assignment to codebook
- Better gradient flow
- More stable training
- Higher generation quality
- **Recommended for most uses**

**KL:**
- Continuous latent space
- Standard VAE approach
- May be easier to understand
- Can be unstable without annealing

### Q: How many epochs should I train for?

**A:** Typical training times:

- **VAE**: 300-500 epochs (6-12 hours on 2x V100)
- **DDPM**: 300-500 epochs (12-24 hours on 2x V100)

Monitor validation metrics and stop when they plateau.

## Multi-GPU Training

### Q: How do I use multiple GPUs?

**A:**

```bash
python train.py --num_gpus 2  # Use 2 GPUs
```

The code automatically handles distributed training with PyTorch DDP.

### Q: Training fails with multiple GPUs

**A:** Common issues:

1. **Port conflict** - change master port:
   ```bash
   export MASTER_PORT=6021
   ```

2. **NCCL errors** - check CUDA/NCCL compatibility
3. **Batch size** - ensure batch_size is divisible by num_gpus
4. **Network issues** - ensure GPUs can communicate

### Q: How do I adjust batch size for multiple GPUs?

**A:** The total batch size = `batch_size * num_gpus`

For effective batch size of 64 with 2 GPUs:
```bash
--num_gpus 2 --opt data.batch_size 32
```

## Evaluation

### Q: How do I evaluate my model?

**A:**

```bash
python train.py \
    --eval \
    --pretrained path/to/checkpoint.pth \
    --ntest 100
```

This computes:
- Chamfer Distance
- Coverage
- Minimum Matching Distance
- Generates sample visualizations

### Q: What are good metric values?

**A:** Typical values on ShapeNet (lower is better for CD):

- **Chamfer Distance**: 0.001 - 0.005 (depends on normalization)
- **Coverage**: 0.4 - 0.6 (higher is better)
- **MMD**: Similar to CD

Values depend on category, number of points, and normalization.

## Configuration

### Q: How do I change configuration?

**A:** Three ways:

1. **YAML file:**
   ```bash
   --config configs/my_config.yaml
   ```

2. **Command line:**
   ```bash
   --opt data.batch_size 64 training.epochs 300
   ```

3. **Combination:**
   ```bash
   --config configs/base.yaml --opt training.opt.lr 5e-5
   ```

### Q: What are the most important hyperparameters?

**A:** Key parameters to tune:

1. **Learning rate** (`training.opt.lr`): 1e-4 is a good starting point
2. **Latent dimension** (`vae.latent_dim`): 256-512
3. **Number of flows** (`vae.n_flows`): 3-5
4. **Batch size** (`data.batch_size`): As large as memory allows
5. **Codebook size** (`vae.softvq.n_e`): 32-128

## Troubleshooting

### Q: Where are the logs saved?

**A:** Logs are saved in:
```
experiments/
└── {training_type}_{quantizer}/
    └── {exp_name}/
        ├── checkpoints/
        ├── config.yml
        ├── train.log
        └── tensorboard/
```

### Q: How do I visualize results?

**A:** Use TensorBoard:

```bash
tensorboard --logdir experiments/
```

Open http://localhost:6006 in your browser.

### Q: Training crashed, can I resume?

**A:** Yes! The code auto-saves snapshots. Simply run the same command again:

```bash
python train.py --config configs/my_config.yaml
```

It will detect the snapshot and resume automatically.

## Performance

### Q: How long does training take?

**A:** Approximate times on 2x V100 GPUs:

- **VAE (chair, 500 epochs)**: 8-10 hours
- **DDPM (chair, 500 epochs)**: 15-18 hours

Times vary with:
- Model size
- Number of points
- Batch size
- Number of GPUs

### Q: How can I speed up sampling?

**A:** For DDPM:

1. **Use fewer timesteps** at inference (DDIM)
2. **Reduce model size** (trade-off with quality)
3. **Use FP16 inference** (requires modification)

### Q: What GPU do I need?

**A:** Recommendations:

- **Minimum**: GTX 1080 Ti (11GB)
- **Recommended**: RTX 3090 (24GB) or V100 (16/32GB)
- **Optimal**: A100 (40/80GB)

## Getting Help

### Q: Where can I get more help?

**A:**

1. **Check documentation**: README.md, ARCHITECTURE.md, this FAQ
2. **Search existing issues**: https://github.com/alifuatsahin/DELIGHT/issues
3. **Open a new issue**: Use the bug report or question template
4. **Read the code**: It's well-documented!

### Q: How can I contribute?

**A:** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Q: I found a bug, what should I do?

**A:** Please open an issue with:
- Clear description
- Steps to reproduce
- Error messages
- Environment details

Use the bug report template: https://github.com/alifuatsahin/DELIGHT/issues/new?template=bug_report.md
