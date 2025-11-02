"""DELIGHT: Deep Compression Latent Diffusion for High-quality 3D Shape Generation.

This package provides implementations of:
- Vector Quantized Variational Autoencoder (VQ-VAE) for 3D point clouds
- Denoising Diffusion Probabilistic Models (DDPM) for latent generation
- Training and evaluation pipelines
"""

__version__ = "0.1.0"
__author__ = "DELIGHT Authors"
__license__ = "MIT"

from . import datasets, models, modules, trainers, utils

__all__ = [
    "datasets",
    "models",
    "modules",
    "trainers",
    "utils",
    "__version__",
]
