# DELIGHT: Deep Compression Latent Diffusion for Generation of High-quality Three-dimensional Shapes

A PyTorch implementation of DELIGHT, a deep learning framework for generating high-quality 3D shapes using compressed latent diffusion.

## Features

- **VAE-based Architecture**: Variational Autoencoder with soft vector quantization for efficient latent representation
- **Flow Matching**: Conditional flow matching for high-quality point cloud generation
- **Flexible Training**: Support for both VAE and prior training modes
- **Distributed Training**: Multi-GPU training support with PyTorch distributed
- **Comprehensive Evaluation**: Built-in metrics for point cloud quality assessment

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alifuatsahin/DELIGHT.git
cd DELIGHT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install third-party libraries:
```bash
cd third_party/ChamferDistancePytorch/chamfer3D
python setup.py install
cd ../../PyTorchEMD
python setup.py install
cd ../..
```

## Usage

### Training

Train a VAE model:
```bash
python train.py --config configs/vae_config.yml --num_gpus 1
```

Train a prior model:
```bash
python train.py --config configs/prior_config.yml --vae_checkpoint path/to/vae/checkpoint.pth
```

### Evaluation

Evaluate a trained model:
```bash
python train.py --eval --pretrained path/to/checkpoint.pth
```

### Generation

Generate point clouds using a trained model:
```bash
python main.py
```

## Project Structure

```
DELIGHT/
├── models/           # Model architectures (VAE, encoder, decoder)
├── modules/          # Building blocks (attention, flows, CFM)
├── trainers/         # Training logic for different model types
├── utils/            # Utility functions (data loading, visualization, metrics)
├── datasets/         # Dataset loaders
├── third_party/      # External dependencies
├── train.py          # Main training script
├── main.py           # Inference and visualization script
└── default_config.py # Default configuration settings
```

## Configuration

The model configuration is managed through YACS config files. Key parameters include:

- `vae.latent_dim`: Latent space dimensionality
- `vae.quantizer`: Type of quantization ('softvq' or 'kl')
- `data.n_sample_points`: Number of points per point cloud
- `training.opt.lr`: Learning rate
- `training.epochs`: Number of training epochs

See `default_config.py` for all available options.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{delight2024,
  title={DELIGHT: Deep Compression Latent Diffusion for Generation of High-quality Three-dimensional Shapes},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Third-party libraries: Chamfer Distance, Earth Mover's Distance, PVCNN
- Built with PyTorch and PyTorch3D

