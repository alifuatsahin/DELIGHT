# DELIGHT Architecture Documentation

This document provides a technical overview of the DELIGHT architecture and implementation details.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Code Structure](#code-structure)
5. [Key Components](#key-components)

## Overview

DELIGHT uses a two-stage approach for 3D shape generation:

1. **Stage 1 (VAE Training)**: Compress 3D point clouds into a discrete latent space using VQ-VAE
2. **Stage 2 (DDPM Training)**: Learn to generate shapes by training a diffusion model in the compressed latent space

## Model Architecture

### VQ-VAE (Variational Autoencoder with Vector Quantization)

```
Point Cloud (N, 3) 
    ↓
Encoder (PVCNN-based)
    ↓
Continuous Features (B, latent_dim)
    ↓
Quantizer (SoftVQ or KL)
    ↓
Discrete Latent Codes (B, latent_dim)
    ↓
Decoder (Normalizing Flow)
    ↓
Reconstructed Point Cloud (N, 3)
```

#### Encoder
- **Architecture**: Point-Voxel CNN (PVCNN)
- **Input**: Point cloud with shape (B, N, 3)
- **Output**: Continuous feature vector (B, latent_dim)
- **Features**: 
  - Efficient point cloud processing with voxelization
  - Multi-scale feature extraction
  - Permutation invariant

#### Quantizer
Two options available:

**1. Soft Vector Quantization (SoftVQ)**
- Uses soft assignment to multiple codebooks
- Parameters:
  - `n_e`: Codebook size per codebook (default: 56)
  - `e_dim`: Embedding dimension (default: 8)
  - `num_codebooks`: Number of codebooks (default: 64)
  - `tau`: Temperature for soft assignment (default: 0.07)
- Benefits: Smoother training, better gradient flow

**2. KL Divergence**
- Standard VAE with Gaussian latent space
- Uses reparameterization trick
- Optional KL annealing for stable training

#### Decoder
- **Architecture**: Normalizing Flow (RQS)
- **Input**: Discrete latent codes (B, latent_dim)
- **Output**: Point cloud parameters
- **Components**:
  - Point Prior Network: Generates point locations
  - Weight Network: Generates mixture weights
  - Flow Layers: Transform base distribution to target distribution
- **Sampling**: Uses mixture of Gaussian distributions

### DDPM (Denoising Diffusion Probabilistic Model)

```
Latent Code (B, latent_dim)
    ↓
Add Noise (forward diffusion)
    ↓
Noisy Latent (B, latent_dim)
    ↓
U-Net Denoiser
    ↓
Predicted Noise/Clean Latent
    ↓
Iterative Denoising (reverse diffusion)
    ↓
Clean Latent Code
```

#### U-Net Denoiser
- **Architecture**: 1D U-Net with attention
- **Features**:
  - Multi-scale processing with skip connections
  - Self-attention at multiple resolutions
  - Time step conditioning
  - Optional xformers for memory-efficient attention

#### Diffusion Process
- **Forward Process**: Gradually add Gaussian noise over T timesteps
- **Reverse Process**: Learn to denoise step by step
- **Parameterization**: Predict noise (eps) or clean data (x0)
- **Schedule**: Linear, cosine, or custom beta schedules

## Training Pipeline

### Stage 1: VAE Training

```python
# Pseudocode
for epoch in range(num_epochs):
    for batch in dataloader:
        # Encode point cloud to latent
        latent, entropy_loss, info = vae.encode(point_cloud)
        
        # Decode latent back to point cloud
        recon_loss, weights = vae.decoder(point_cloud, latent)
        
        # Total loss
        loss = recon_loss + kl_weight * entropy_loss
        
        # Optimize
        loss.backward()
        optimizer.step()
```

**Key Hyperparameters:**
- Learning rate: 1e-4 (with warmup and decay)
- Batch size: 32 (per GPU)
- Epochs: 500
- Optimizer: AdamW with EMA
- Loss: Point cloud reconstruction + quantization loss

### Stage 2: DDPM Training

```python
# Pseudocode
vae = load_pretrained_vae()
vae.eval()

for epoch in range(num_epochs):
    for batch in dataloader:
        # Get latent codes from VAE
        with torch.no_grad():
            latent = vae.encode(point_cloud)
        
        # Sample random timestep
        t = sample_timestep()
        
        # Add noise to latent
        noisy_latent = add_noise(latent, t)
        
        # Predict noise
        predicted_noise = ddpm(noisy_latent, t)
        
        # MSE loss
        loss = mse_loss(predicted_noise, true_noise)
        
        # Optimize
        loss.backward()
        optimizer.step()
```

**Key Hyperparameters:**
- Learning rate: 1e-4
- Batch size: 32
- Epochs: 500
- Timesteps: 1000
- Beta schedule: Linear

## Code Structure

```
DELIGHT/
├── models/
│   ├── vae.py              # VQ-VAE model
│   ├── ddpm.py             # DDPM model
│   ├── encoder.py          # Point cloud encoder
│   ├── decoder.py          # Flow-based decoder
│   └── quantizers/
│       ├── softvq.py       # Soft VQ quantizer
│       └── kl.py           # KL divergence quantizer
│
├── modules/
│   ├── flows.py            # Normalizing flow layers (RQS)
│   ├── layers.py           # Custom neural network layers
│   ├── pvcnn2.py           # Point-Voxel CNN implementation
│   └── fre_loss.py         # Frequency-based loss
│
├── datasets/
│   ├── dataset.py          # ShapeNet dataset loader
│   ├── data_path.py        # Data path configuration
│   └── preprocessing.py    # Data preprocessing utilities
│
├── trainers/
│   ├── base_trainer.py     # Base trainer class
│   ├── vae_trainer.py      # VAE training logic
│   └── ddpm_trainer.py     # DDPM training logic
│
├── utils/
│   ├── utils.py            # General utilities
│   ├── vis_helper.py       # Visualization helpers
│   ├── eval_metrics.py     # Evaluation metrics (CD, EMD)
│   ├── ema.py              # Exponential Moving Average
│   └── diffusion_helper.py # Diffusion utilities
│
└── third_party/
    ├── ChamferDistancePytorch/  # Chamfer Distance
    ├── PyTorchEMD/              # Earth Mover's Distance
    ├── pvcnn/                   # PVCNN operators
    └── torchdiffeq/             # ODE solvers
```

## Key Components

### 1. Point-Voxel CNN (PVCNN)

Combines the efficiency of voxel-based convolutions with the accuracy of point-based methods:
- **Voxelization**: Converts point clouds to voxel grids
- **3D Convolutions**: Efficient feature extraction
- **Devoxelization**: Projects features back to points
- **Set Abstraction**: PointNet++-style aggregation

### 2. Normalizing Flows

Transforms simple base distributions to complex target distributions:
- **Rational Quadratic Splines (RQS)**: Flexible transformations
- **Invertible**: Can compute both forward and inverse
- **Tractable Jacobian**: For exact likelihood computation

### 3. Soft Vector Quantization

Improves upon traditional VQ-VAE:
- **Soft Assignment**: Differentiable quantization
- **Multiple Codebooks**: Increased capacity
- **Temperature Annealing**: Gradual sharpening of assignments
- **Entropy Regularization**: Encourages codebook usage

### 4. Diffusion Process

Generates high-quality samples through iterative refinement:
- **Noise Schedule**: Controls noise level at each step
- **U-Net Denoiser**: Predicts noise at each timestep
- **DDIM Sampling**: Faster sampling with fewer steps

## Training Tips

### Memory Optimization
- Use gradient checkpointing: `ddpm.use_checkpoint = True`
- Reduce batch size if OOM
- Use xformers attention: `ddpm.use_xformers_attention = True`

### Convergence
- Use EMA for stable model weights
- KL annealing for VAE training
- Learning rate warmup and decay
- Monitor codebook usage in SoftVQ

### Evaluation
- **Chamfer Distance (CD)**: Measures reconstruction quality
- **Earth Mover's Distance (EMD)**: Alternative distance metric
- **Coverage**: Measures diversity of generated samples
- **Minimum Matching Distance (MMD)**: Measures fidelity

## Configuration Guide

See `configs/example_config.yaml` for a complete configuration template.

### Critical Parameters

**VAE:**
- `latent_dim`: Higher = more capacity, slower training
- `n_flows`: More flows = better flexibility
- `softvq.num_codebooks`: Trade-off between capacity and speed

**DDPM:**
- `timesteps`: More = better quality, slower sampling
- `beta_schedule`: Controls noise addition rate
- `use_xformers_attention`: Enables memory-efficient attention

**Data:**
- `n_sample_points`: More points = better detail, higher memory
- `batch_size`: Trade-off between speed and memory

## References

- Point-Voxel CNN: [MIT Han Lab](https://github.com/mit-han-lab/pvcnn)
- Diffusion Models: [DDPM Paper](https://arxiv.org/abs/2006.11239)
- Vector Quantization: [VQ-VAE-2](https://arxiv.org/abs/1906.00446)
- Normalizing Flows: [Neural Spline Flows](https://arxiv.org/abs/1906.04032)
