# DELIGHT: Design Exploration via Latent Inference for Generating High-quality Topologies

A state-of-the-art generative model for high-quality 3D point cloud generation using Variational Autoencoders (VAE) with advanced quantization and conditional flow matching techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Configuration](#configuration)
- [Models](#models)
- [Evaluation](#evaluation)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🎯 Overview

DELIGHT is a generative model designed to create high-quality 3D point clouds through a two-stage pipeline:

1. **VAE Training Stage**: Learns a compressed latent representation of 3D shapes using a Variational Autoencoder with soft vector quantization
2. **Prior Training Stage**: Learns the prior distribution over the latent codes using conditional flow matching (CFM)

The model combines:
- **PointNet++ Architecture**: For efficient point cloud encoding/decoding
- **Soft Vector Quantization**: For discrete, interpretable latent representations
- **Conditional Flow Matching**: For generating diverse and high-quality samples
- **Multi-head Attention**: For capturing complex geometric relationships

## ✨ Features

- **Efficient Compression**: Reduces 3D shapes to compact latent representations
- **High-Quality Generation**: Generates diverse point clouds with fine geometric details
- **Flexible Architecture**: Supports multiple quantization methods (SoftVQ, KL)
- **Advanced Training**: Includes KL annealing, EMA updates, and cosine annealing scheduling
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Configurable Pipeline**: Extensive configuration options via YAML and Python configs
- **Visualization Tools**: Built-in rendering and visualization utilities

## 📦 Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (13.x recommended for PVCNN backend)
- PyTorch 2.0+
- NVIDIA Build Tools (for C++/CUDA extensions)

### Step 1: Install Build Tools

**On Windows:**
1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Select "Desktop development with C++" workload
3. Include MSVC and Windows 10 SDK components

**On Linux:**
```bash
sudo apt-get install build-essential
```

### Step 2: Set Up Python Environment

```bash
# Using conda (recommended)
conda create -n delight python=3.11
conda activate delight

# Install CUDA toolkit
conda install -c nvidia cudatoolkit=11.8

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3: Install Dependencies

```bash
cd DELIGHT
pip install -r requirements.txt
```

**Key dependencies:**
- torch & torchvision
- torchdiffeq (for ODE solving)
- loguru (for logging)
- yacs (for configuration management)
- ninja (for C++ extension compilation)
- mitsuba (for rendering - optional)

### Step 4: Compile CUDA Extensions

The CUDA extensions will be automatically compiled on first run. Ensure the C++ compiler is available:

```bash
# Verify CUDA toolkit
nvcc --version

# Verify C++ compiler
cl  # Windows
gcc --version  # Linux
```

## 📁 Project Structure

```
DELIGHT/
├── data/                           # Data directory
│   ├── sampling.py                 # Point cloud sampling utilities
│   ├── ShapeNetCore.v2/            # ShapeNet dataset
│   └── ShapeNetCore.v2.PC15k/      # Pre-sampled point clouds
├── models/                         # Core model components
│   ├── vae.py                      # Main VAE model
│   ├── encoder.py                  # Point cloud encoder
│   ├── decoder.py                  # Point cloud decoder
│   ├── fm_prior.py                 # Flow matching prior
│   ├── surrogate.py                # Surrogate model for evaluation
│   └── quantizers/
│       ├── softvq.py               # Soft vector quantization
│       └── kl.py                   # KL-based quantization
├── modules/                        # Neural network modules
│   ├── pvcnn.py                    # PVCNN architecture
│   ├── attention.py                # Attention mechanisms
│   ├── cfm.py                      # Conditional flow matching
│   ├── flows.py                    # Flow-based models
│   └── layers.py                   # Custom layers
├── datasets/                       # Data loading utilities
│   ├── dataset.py                  # Dataset classes
│   ├── preprocessing.py            # Data preprocessing
│   └── __init__.py
├── trainers/                       # Training logic
│   ├── base_trainer.py             # Base trainer class
│   ├── vae_trainer.py              # VAE trainer
│   ├── prior_trainer.py            # Prior trainer
│   └── surrogate_trainer.py        # Surrogate trainer
├── third_party/                    # Third-party libraries
│   ├── pvcnn/                      # PVCNN implementation
│   ├── PyTorchEMD/                 # Earth Mover Distance
│   ├── ChamferDistancePytorch/     # Chamfer distance
│   └── torchdiffeq/                # Differentiable ODE solver
├── serialization/                  # Point cloud serialization
│   ├── default.py                  # Default serialization
│   ├── hilbert.py                  # Hilbert curve order
│   └── z_order.py                  # Z-order curve order
├── utils/                          # Utility functions
│   ├── eval_helper.py              # Evaluation utilities
│   ├── eval_metrics.py             # Evaluation metrics
│   ├── data_helper.py              # Data utilities
│   ├── vis_helper.py               # Visualization
│   ├── render_mitsuba_pc.py        # Rendering point clouds
│   └── utils.py                    # General utilities
├── scripts/                        # Training scripts
│   ├── train_vae.sh                # VAE training script
│   ├── train_prior.sh              # Prior training script
│   ├── test.sh                     # Testing script
│   ├── resume.sh                   # Resume training
│   └── preprocess.sh               # Data preprocessing
├── main.py                         # Inference/sampling script
├── train.py                        # Main training entry point
├── default_config.py               # Default configuration
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Training a VAE Model

```bash
cd DELIGHT
bash scripts/train_vae.sh
```

This will:
- Load the dataset from `data/ShapeNetCore.v2/`
- Train a VAE with SoftVQ quantization
- Save checkpoints and logs to `experiments/`

### 2. Training a Prior Model

```bash
bash scripts/train_prior.sh
```

This will:
- Load a pre-trained VAE checkpoint
- Train a conditional flow matching prior
- Enable generation of diverse samples

### 3. Sampling/Inference

```bash
python main.py
```

This script will:
- Load trained VAE and prior models
- Generate point cloud samples
- Render samples to images

## 📊 Data Preparation

### ShapeNet Dataset

1. **Download ShapeNet Core v2** from the [official website](https://shapenet.org/)

2. **Organize the data:**
```
data/
├── ShapeNetCore.v2/
│   ├── 02691155/  # airplane
│   ├── 02808440/  # bathtub
│   ├── 03001627/  # chair
│   └── ...
```

3. **Preprocess point clouds** (optional, pre-sampled version provided):
```bash
bash scripts/preprocess.sh
```

### Dataset Categories

The dataset supports 45 object categories including:
- `chair`, `airplane`, `car`, `table`, `guitar`, `lamp`, `monitor`, `bottle`, `cabinet`, `sofa`, `train`, `telephone`, etc.

## 🎓 Training

### Configuration

Training is controlled via [default_config.py](default_config.py). Key configurations:

**VAE Configuration:**
```python
cfg.vae.latent_dim = 512           # Latent dimension
cfg.vae.quantizer = 'softvq'       # 'softvq' or 'kl'
cfg.vae.input_dim = 3              # Input dimension (3 for XYZ)
```

**Data Configuration:**
```python
cfg.data.dataset = 'ShapeNetCore.v2'
cfg.data.categories = 'chair'      # Can be list or string
cfg.data.n_sample_points = 2048    # Points per cloud
cfg.data.batch_size = 32
```

**Training Configuration:**
```python
cfg.training.epochs = 500
cfg.training.type = "vae"          # 'vae' or 'prior'
cfg.training.opt.lr = 0.0001
cfg.training.opt.scheduler = 'cosine_anneal_nocycle'
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py
```

### Resume Training

```bash
bash scripts/resume.sh
```

Or manually:
```bash
python train.py --pretrained <checkpoint_path>
```

## 🧠 Models

### VAE Architecture

**Encoder:**
- PointNet++ with set abstraction layers
- Progressively downsamples point clouds
- Outputs feature vectors for quantization

**Quantizer (SoftVQ):**
- Multiple learnable codebooks (default: 32)
- Temperature-based soft selection
- Entropy regularization for codebook usage

**Decoder:**
- Diffusion-based decoding
- Conditional flow matching
- Generates high-quality point clouds from latent codes

### Flow Matching Prior

- Learns prior distribution over latent codes
- Supports:
  - Optimal Transport (OT) method
  - Conditional flow matching
  - Hybrid coupling layers
  - Multi-head attention

## 📈 Evaluation

### Metrics

The framework computes standard 3D generation metrics:
- **Chamfer Distance (CD)**: Geometric similarity
- **Earth Mover Distance (EMD)**: Point distribution similarity
- **Coverage**: Percentage of test set covered
- **Density**: Point density of generated clouds

### Evaluation Script

```bash
bash scripts/test.sh
```

Or programmatically:
```python
from utils.eval_metrics import compute_metrics

metrics = compute_metrics(
    generated_pcs,
    reference_pcs,
    metrics=['cd', 'emd', 'coverage', 'density']
)
```

## 💻 Usage Examples

### Generate Samples

```python
from models.vae import VAE
from default_config import cfg
import torch

# Load config and model
cfg.merge_from_file("path/to/config.yml")
vae = VAE(cfg).to('cuda')
checkpoint = torch.load("path/to/checkpoint.pth")
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

# Generate samples
with torch.no_grad():
    batch_size = 4
    latent = torch.randn(batch_size, cfg.vae.latent_dim).to('cuda')
    samples = vae.decode(latent, n_sampled_points=2048)
    # samples shape: (batch_size, 2048, 3)
```

### Encode Point Clouds

```python
# Load point clouds (batch_size, num_points, 3)
point_clouds = torch.randn(4, 2048, 3).to('cuda')

# Encode to latent space
latent, entropy_loss, info = vae.encode(point_clouds)
# latent shape: (batch_size, latent_dim)
```

### Reconstruction

```python
# Reconstruct point clouds
reconstructed, recon_loss = vae.reconstruct(point_clouds)
# reconstructed shape: (batch_size, num_points, 3)
```

### Interpolation

```python
# Interpolate between two point clouds
z1, _, _ = vae.encode(pc1)
z2, _, _ = vae.encode(pc2)

# Linear interpolation in latent space
alphas = torch.linspace(0, 1, 10)
interpolated = []
for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2
    pc_interp = vae.decode(z_interp)
    interpolated.append(pc_interp)
```

## 🔧 Troubleshooting

### CUDA Extension Compilation Errors

**Error:** `subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1`

**Solution:**
1. Install Microsoft C++ Build Tools
2. Open "Developer Command Prompt for Visual Studio"
3. Activate your conda environment
4. Verify: `cl` and `nvcc --version`
5. Re-run training

### Out of Memory (OOM) Errors

**Solutions:**
- Reduce batch size: `cfg.data.batch_size = 16`
- Reduce number of points: `cfg.data.n_sample_points = 1024`
- Reduce model size: `cfg.vae.latent_dim = 256`
- Enable gradient checkpointing (if supported)

### Dataset Not Found

**Solution:**
- Ensure ShapeNet data is in `data/ShapeNetCore.v2/`
- Verify symlinks are correct
- Check category ID (e.g., `03001627` for chairs)

### Poor Generation Quality

**Potential causes and fixes:**
- Insufficient training: Increase `cfg.training.epochs`
- Incorrect data normalization: Check `cfg.data.normalize_global`
- Bad prior: Re-train prior model with more epochs
- Hyperparameter tuning: Adjust learning rate, KL weight

## 📚 References

- **PointNet++**: [Original Paper](https://arxiv.org/abs/1706.02413)
- **Vector Quantized Autoencoders**: [VQ-VAE](https://arxiv.org/abs/1711.00937)
- **Conditional Flow Matching**: [CFM Paper](https://arxiv.org/abs/2302.00482)
- **Optimal Transport**: [OT Planning](https://arxiv.org/abs/2106.05933)
- **ShapeNet Dataset**: [Official Website](https://shapenet.org/)

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email:** alifuatpc@gmail.com
- **GitHub:** [alifuatsahin](https://github.com/alifuatsahin)

---