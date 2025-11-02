# DELIGHT: Deep ComprEssion Latent dIffusion for Generation of High-quality Three-dimensional shapes

A PyTorch implementation of DELIGHT, a method for high-quality 3D shape generation using compressed latent diffusion models with Vector Quantized Variational Autoencoders (VQ-VAE) and Denoising Diffusion Probabilistic Models (DDPM).

📚 **[Complete Documentation](DOCUMENTATION.md)** | 🚀 **[Quick Start](QUICKSTART.md)** | 🏗️ **[Architecture](ARCHITECTURE.md)** | ❓ **[FAQ](FAQ.md)**

## Overview

DELIGHT combines:
- **VQ-VAE with Soft Vector Quantization**: Efficient compression of 3D point clouds into discrete latent representations
- **Latent Diffusion Models**: Generation of high-quality 3D shapes in the compressed latent space
- **Normalizing Flows**: Flexible prior and posterior modeling for improved generation quality

## Features

- ✨ High-quality 3D point cloud generation
- 🚀 Efficient latent space compression with SoftVQ
- 🎨 Support for multiple shape categories from ShapeNet
- 📊 Comprehensive training and evaluation pipeline
- 🔧 Flexible configuration system
- 💾 Checkpoint saving and resumption
- 📈 TensorBoard logging and visualization

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/alifuatsahin/DELIGHT.git
cd DELIGHT
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Compile third-party CUDA extensions

The repository includes several third-party modules that require compilation:

#### ChamferDistancePytorch
```bash
cd third_party/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../../..
```

#### PyTorchEMD (Earth Mover's Distance)
```bash
cd third_party/PyTorchEMD
python setup.py install
cd ../..
```

**Note**: These extensions require CUDA and a compatible C++ compiler. Make sure your CUDA_HOME environment variable is set correctly.

## Dataset Preparation

DELIGHT uses the ShapeNet dataset. Follow these steps to prepare your data:

1. Download ShapeNet v2 from the [official website](https://shapenet.org/)
2. Process the dataset into point cloud format (.npy files)
3. Organize the data structure as follows:

```
ShapeNet/
├── 02691156/  # airplane
│   ├── train/
│   │   ├── <model_id>.npy
│   │   └── ...
│   ├── val/
│   └── test/
├── 03001627/  # chair
│   └── ...
└── ...
```

Each .npy file should contain a point cloud with shape (N, 3), where N is the number of points (typically 15000).

### Data Preprocessing Script

```bash
# Example preprocessing (customize based on your data format)
bash scripts/preprocess.sh
```

## Usage

### Training a VQ-VAE Model

Train a VQ-VAE to compress point clouds into discrete latent codes:

```bash
# Single GPU training
python train.py \
    --num_gpus 1 \
    --opt \
    data.categories chair \
    data.batch_size 32 \
    training.type vae \
    vae.quantizer softvq \
    vae.latent_dim 512

# Multi-GPU training
python train.py --num_gpus 2 \
    --opt data.categories airplane data.batch_size 32
```

Or use the provided training script:

```bash
bash scripts/train_vae.sh
```

### Training a Diffusion Prior (DDPM)

After training the VQ-VAE, train a diffusion model in the latent space:

```bash
python train.py \
    --num_gpus 2 \
    --vae_checkpoint <path_to_vae_checkpoint> \
    --opt \
    training.type ddpm \
    data.batch_size 32
```

Or use the provided script:

```bash
# Edit the CKPT path in the script first
bash scripts/train_prior.sh
```

### Evaluation and Sampling

Generate samples and evaluate the trained model:

```bash
python train.py \
    --eval \
    --pretrained <path_to_checkpoint> \
    --ntest 100
```

### Configuration

The model supports extensive configuration through:
1. **YAML config files** (in `configs/`)
2. **Command-line arguments** with `--opt`
3. **Default configurations** (in `default_config.py`)

Key configuration options:

```yaml
vae:
  latent_dim: 512              # Latent dimension
  quantizer: softvq            # 'softvq' or 'kl'
  n_flows: 4                   # Number of normalizing flow layers
  
  softvq:
    n_e: 56                    # Codebook size per codebook
    e_dim: 8                   # Embedding dimension
    num_codebooks: 64          # Number of codebooks
    
ddpm:
  timesteps: 1000              # Diffusion timesteps
  beta_schedule: linear        # 'linear', 'cosine', etc.
  loss_type: l2                # 'l1' or 'l2'
  
data:
  categories: chair            # ShapeNet category
  n_sample_points: 2048        # Points per shape
  batch_size: 32               # Batch size
  
training:
  epochs: 500                  # Training epochs
  opt:
    lr: 0.0001                 # Learning rate
    ema: True                  # Use EMA
```

## Project Structure

```
DELIGHT/
├── configs/              # Configuration files
├── datasets/             # Dataset loaders and preprocessing
├── models/               # Model architectures
│   ├── vae.py           # VQ-VAE model
│   ├── ddpm.py          # Diffusion model
│   ├── encoder.py       # Point cloud encoder
│   ├── decoder.py       # Point cloud decoder
│   └── quantizers/      # Quantization modules
├── modules/              # Reusable modules (flows, layers)
├── trainers/             # Training logic
├── utils/                # Utilities (visualization, metrics)
├── third_party/          # Third-party dependencies
├── scripts/              # Training and preprocessing scripts
├── main.py               # Simple inference example
└── train.py              # Main training script
```

## Results

DELIGHT achieves state-of-the-art results on 3D shape generation:
- High-quality point cloud generation
- Efficient compression with VQ-VAE
- Fast sampling with latent diffusion

## Troubleshooting

### CUDA Issues

If you encounter CUDA compilation errors:
1. Ensure CUDA toolkit is installed: `nvcc --version`
2. Set CUDA_HOME: `export CUDA_HOME=/usr/local/cuda`
3. Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

### Out of Memory

If you run out of GPU memory:
- Reduce batch size: `--opt data.batch_size 16`
- Reduce number of points: `--opt data.n_sample_points 1024`
- Use gradient checkpointing: `--opt ddpm.use_checkpoint True`

### Data Loading Issues

If data loading is slow or crashes:
- Reduce number of workers: `--opt data.num_workers 4`
- Check data path configuration in `datasets/data_path.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delight2024,
  title={DELIGHT: Deep Compression Latent Diffusion for Generation of High-quality Three-dimensional shapes},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Acknowledgements

This project builds upon several excellent works:
- [Point-Voxel CNN (PVCNN)](https://github.com/mit-han-lab/pvcnn) for efficient point cloud processing
- [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion) for the diffusion framework
- [PointFlow](https://github.com/stevenygd/PointFlow) for dataset processing utilities
- [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) for evaluation metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: Third-party components in the `third_party/` directory are subject to their respective licenses.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
