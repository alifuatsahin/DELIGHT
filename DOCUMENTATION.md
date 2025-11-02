# DELIGHT Documentation Index

Welcome to the DELIGHT documentation! This page provides an overview of all available documentation and guides.

## 📚 Documentation Structure

### Getting Started

- **[README.md](README.md)** - Main project overview, installation, and basic usage
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide to get running in 30 minutes
- **[FAQ.md](FAQ.md)** - Frequently asked questions and troubleshooting

### Technical Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed architecture and implementation guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### Configuration

- **[configs/example_config.yaml](configs/example_config.yaml)** - Complete configuration example
- **[default_config.py](default_config.py)** - Default configuration definitions

### Community

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines
- **[LICENSE](LICENSE)** - MIT License terms

## 🚀 Quick Navigation

### I want to...

#### Install and Setup
→ Start with [QUICKSTART.md](QUICKSTART.md) for fastest setup  
→ Or see [README.md](README.md) Installation section for detailed instructions

#### Train a Model
→ See [README.md](README.md) Usage section  
→ Check [configs/example_config.yaml](configs/example_config.yaml) for configuration  
→ Review [FAQ.md](FAQ.md) for common training issues

#### Understand the Architecture
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) for technical details  
→ Check code comments in `models/` directory

#### Troubleshoot Issues
→ Check [FAQ.md](FAQ.md) first  
→ Search [existing issues](https://github.com/alifuatsahin/DELIGHT/issues)  
→ Open a [new issue](https://github.com/alifuatsahin/DELIGHT/issues/new)

#### Contribute
→ Read [CONTRIBUTING.md](CONTRIBUTING.md)  
→ Review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)  
→ Check [issue templates](.github/ISSUE_TEMPLATE/)

## 📖 Documentation by Topic

### Installation
- [System Requirements](README.md#requirements)
- [Dependencies](requirements.txt)
- [CUDA Extensions](README.md#installation)
- [Troubleshooting](FAQ.md#installation-issues)

### Data Preparation
- [Dataset Setup](README.md#dataset-preparation)
- [Custom Datasets](FAQ.md#can-i-use-my-own-dataset)
- [Data Configuration](ARCHITECTURE.md#key-components)

### Training
- [VAE Training](README.md#training-a-vq-vae-model)
- [DDPM Training](README.md#training-a-diffusion-prior-ddpm)
- [Configuration Options](configs/example_config.yaml)
- [Training Tips](ARCHITECTURE.md#training-tips)
- [Multi-GPU Training](FAQ.md#multi-gpu-training)

### Evaluation
- [Generating Samples](README.md#evaluation-and-sampling)
- [Metrics](ARCHITECTURE.md#evaluation)
- [Visualization](FAQ.md#how-do-i-visualize-results)

### Model Architecture
- [Overview](ARCHITECTURE.md#overview)
- [VQ-VAE Details](ARCHITECTURE.md#vq-vae-variational-autoencoder-with-vector-quantization)
- [DDPM Details](ARCHITECTURE.md#ddpm-denoising-diffusion-probabilistic-model)
- [Code Structure](ARCHITECTURE.md#code-structure)

### Configuration
- [Example Config](configs/example_config.yaml)
- [Configuration Guide](ARCHITECTURE.md#configuration-guide)
- [Important Parameters](FAQ.md#what-are-the-most-important-hyperparameters)

### Troubleshooting
- [Common Issues](FAQ.md)
- [CUDA Problems](FAQ.md#chamferdistance-compilation-fails)
- [Training Issues](FAQ.md#training-issues)
- [Performance](FAQ.md#performance)

### Development
- [Code Style](CONTRIBUTING.md#code-style-guidelines)
- [Testing](CONTRIBUTING.md#testing)
- [Documentation Standards](CONTRIBUTING.md#documentation)
- [Pull Request Process](CONTRIBUTING.md#pull-requests)

## 🔧 API Documentation

### Core Models
```python
from models import VAE, DDPM

# VAE for point cloud compression
vae = VAE(config)
samples, labels = vae.sample(n_sampled_points=2048, n_samples=10)

# DDPM for latent generation
ddpm = DDPM(config)
output = ddpm(latent, timesteps)
```

See inline code documentation for detailed API usage.

### Key Classes
- `models.VAE` - Vector Quantized VAE ([models/vae.py](models/vae.py))
- `models.DDPM` - Diffusion model ([models/ddpm.py](models/ddpm.py))
- `models.Encoder` - Point cloud encoder ([models/encoder.py](models/encoder.py))
- `models.Decoder` - Flow-based decoder ([models/decoder.py](models/decoder.py))
- `trainers.VAETrainer` - VAE training logic ([trainers/vae_trainer.py](trainers/vae_trainer.py))
- `trainers.DDPMTrainer` - DDPM training logic ([trainers/ddpm_trainer.py](trainers/ddpm_trainer.py))

## 📊 Examples

### Training Examples
See [scripts/](scripts/) directory for example training scripts:
- `train_vae.sh` - VAE training
- `train_prior.sh` - DDPM training
- `test.sh` - Evaluation
- `resume.sh` - Resume training

### Configuration Examples
- [configs/example_config.yaml](configs/example_config.yaml) - Complete example
- [configs/default.yaml](configs/default.yaml) - Minimal example

## 🔗 External Resources

### Papers and References
- DDPM: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- VQ-VAE: [Vector Quantized Variational Autoencoders](https://arxiv.org/abs/1711.00937)
- PVCNN: [Point-Voxel CNN for Efficient 3D Deep Learning](https://arxiv.org/abs/1907.03739)

### Related Projects
- [PointFlow](https://github.com/stevenygd/PointFlow) - Normalizing flows for point clouds
- [PVCNN](https://github.com/mit-han-lab/pvcnn) - Efficient point cloud processing
- [Guided Diffusion](https://github.com/openai/guided-diffusion) - Diffusion models

### Datasets
- [ShapeNet](https://shapenet.org/) - Large-scale 3D shape repository
- [ModelNet](https://modelnet.cs.princeton.edu/) - Alternative 3D dataset

## 💬 Community and Support

### Getting Help
1. Check this documentation
2. Search [FAQ.md](FAQ.md)
3. Look through [existing issues](https://github.com/alifuatsahin/DELIGHT/issues)
4. Ask a question using the [question template](.github/ISSUE_TEMPLATE/question.md)

### Reporting Issues
- **Bugs**: Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- **Features**: Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Pull request process
- Testing requirements

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🎯 Recommended Reading Order

### For Users
1. [README.md](README.md) - Overview
2. [QUICKSTART.md](QUICKSTART.md) - Get started
3. [configs/example_config.yaml](configs/example_config.yaml) - Configure your training
4. [FAQ.md](FAQ.md) - Troubleshooting

### For Developers
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Understand the system
2. [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines
3. Code documentation in source files
4. [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community standards

### For Researchers
1. [ARCHITECTURE.md](ARCHITECTURE.md) - Technical details
2. [README.md](README.md) - Reproducibility
3. Source code - Implementation details

---

**Last Updated**: 2024-11-02  
**Version**: 0.1.0

For the most up-to-date documentation, visit: https://github.com/alifuatsahin/DELIGHT
