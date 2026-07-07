# DELIGHT: Design Exploration via Latent Inference for Generating High-quality Topologies

A generative model for high-quality 3D point cloud generation using Variational Autoencoders (VAE) with advanced quantization and conditional flow matching techniques.

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
2. **Prior Training Stage**: Learns the prior distribution over the latent codes using flow matching (FM)

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

> **Linux only.** The CUDA/C++ extensions under `third_party/` are
> JIT-compiled on first import and that build is only exercised on Linux, and
> distributed training uses the NCCL backend, which Windows does not provide.

### Prerequisites

- Linux with an NVIDIA GPU
- Python 3.11
- CUDA 12.8
- PyTorch 2.11.0
- Miniconda or Anaconda
- A host C/C++ toolchain: `sudo apt-get install build-essential`

### Step 1: Install Miniconda

Download and install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for Linux.

### Step 2: Set Up Python Environment

```bash
# Using conda with environment.yml (recommended)
cd DELIGHT
conda env create -f environment.yml
conda activate delight
```

**Key dependencies installed:**
- Python 3.11
- PyTorch 2.11.0 with CUDA 12.8 support (from the `cu128` wheel index)
- `cuda-toolkit` 12.8 — supplies `nvcc`, needed to *build* the extensions below
- torchdiffeq (for ODE solving)
- loguru (for logging)
- ninja (for C++ extension compilation)
- tensorboard, scikit-learn, pandas, scipy
- geomloss + pykeops (only used by the `ot` / `schrodinger_bridge` couplings)
- mitsuba 3.8.0 (optional, for rendering)
- Additional utilities: open3d, trimesh, pymeshlab, and more

### Step 3: Verify Installation

Verify that your CUDA setup is correct:

```bash
# Verify CUDA toolkit
nvcc --version

# Verify PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 4: CUDA extensions

Three vendored extensions are **JIT-compiled on first import**, not at install
time — there is no separate build step to run:

| Extension | Used by |
|---|---|
| `third_party/pvcnn` | encoder/decoder set-abstraction layers (every run) |
| `third_party/PyTorchEMD` | Earth Mover Distance in evaluation |
| `third_party/ChamferDistancePytorch` | Chamfer distance in evaluation |

The first run therefore spends a few minutes in `ninja` before training
starts. Compiled objects are cached under `~/.cache/torch_extensions`, so
later runs start immediately. If the build fails, it is almost always a
missing `nvcc` or a `gcc` too new for the CUDA release — see
[Troubleshooting](#-troubleshooting).

## 📁 Project Structure

```
DELIGHT/
├── models/                         # Core model components
│   ├── vae.py                      # Main VAE model
│   ├── encoder.py                  # Point cloud encoder
│   ├── decoder.py                  # Point cloud decoder
│   ├── fm_prior.py                 # Flow matching prior
│   └── quantizers/
│       ├── softvq.py               # Soft vector quantization
│       └── kl.py                   # KL-based quantization
├── modules/                        # Neural network modules
│   ├── pvcnn.py                    # PVCNN architecture
│   ├── attention.py                # Attention mechanisms
│   ├── cfm.py                      # Conditional flow matching
│   ├── flows.py                    # Flow-based models
│   └── layers.py                   # Custom layers
├── pyansys/                        # Ansys simulation scripts (Licensed ANSYS 25.2+ required)
│   ├── run_sim.py                  # Simulation script
├── datasets/                       # Data loading utilities
│   ├── dataset.py                  # Dataset classes
│   ├── preprocessing.py            # Data preprocessing
│   ├── pasta_equations.ipynb       # Mathematical equations for pasta shapes
│   └── __init__.py
├── trainers/                       # Training logic
│   ├── base_trainer.py             # Base trainer class
│   ├── vae_trainer.py              # VAE trainer
│   └── prior_trainer.py            # Prior trainer
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
├── LICENSE                         # LICENSE file
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Training a VAE Model

```bash
cd DELIGHT
# Edit DATA= at the top of the script to pick your category first.
bash scripts/train_vae.sh
```

This will:
- Load the dataset from `data/ShapeNetCore.v2.PC15k/<synsetid>/` (see
  [Data Preparation](#-data-preparation))
- Train a VAE with SoftVQ quantization
- Save checkpoints and logs to `../experiments/vae_softvq/<exp_name>/`
  (note: `--exp_root` defaults to `../experiments`, i.e. *beside* the repo)

`scripts/train_vae.sh` ships with `DATA="pasta"`, a custom category that is
not part of ShapeNet. Set it to a category you actually have — `chair`,
`airplane`, `car`, … — before the first run.

### 2. Training a Prior Model

```bash
CKPT=../experiments/vae_softvq/<exp>/checkpoints/best_eval.pth \
    bash scripts/train_prior.sh
```

This will:
- Load the pre-trained VAE named by `CKPT`, apply its EMA weights, and freeze it
- Inherit the VAE run's `config.yml`, so the two stages cannot drift apart
- Train a conditional flow matching prior over the frozen latents
- Save to `../experiments/prior/<exp_name>/`

`CKPT` is required; the script exits if it is unset or missing.

### 3. Sampling/Inference

```bash
python main.py --pretrained ../experiments/prior/<exp>/checkpoints/best_eval.pth \
    --num_samples 64 --render --metrics
```

This script will:
- Load a trained prior checkpoint (it embeds the frozen VAE)
- Sample latents from the prior and decode them into point clouds
- Save the samples to `<exp>/samples/`, optionally rendering a preview grid
  (`--render`) and scoring them against the eval split (`--metrics`)

Point `--pretrained` at a **prior** checkpoint. A VAE checkpoint also loads,
but its samples come from codes drawn uniformly at random from the SoftVQ
codebooks, which is a sanity check and not a trained generative prior.

## 📊 Data Preparation

### ShapeNet Dataset

1. **Download ShapeNet Core v2** from the [official website](https://shapenet.org/)

2. **Preprocess the meshes into point clouds:**
```bash
bash scripts/preprocess.sh   # set DATA_DIR / CATEGORIES inside the script
```

3. **Resulting layout.** The loader reads pre-sampled 15k-point clouds, one
   `.npy` of shape `(15000, 3)` per shape, split into `train`/`val`/`test`
   directories under each synset id:

```
data/
└── ShapeNetCore.v2.PC15k/
    ├── 03001627/          # chair
    │   ├── train/  <model_id>.npy ...
    │   ├── val/    <model_id>.npy ...
    │   └── test/   <model_id>.npy ...
    ├── 02691156/          # airplane
    └── ...
```

The path is resolved by `get_path()` in
[datasets/dataset.py](datasets/dataset.py), which looks for
`./data/ShapeNetCore.v2.PC15k/` relative to the directory you launch from —
so run the scripts from the repository root. Point it elsewhere by editing
that function. Raw `ShapeNetCore.v2/` meshes are the *input* to
`preprocess.sh`, not something the trainer reads.

### Dataset Categories

58 category names map to synset ids in `synsetid_to_cate`
([datasets/dataset.py](datasets/dataset.py)) — `chair`, `airplane`, `car`,
`table`, `guitar`, `lamp`, `monitor`, `bottle`, `cabinet`, `sofa`, `train`,
`telephone` and so on. Set one (or `all`) via `data.categories`; you only
need the synsets you actually train on to be present on disk.

Two entries are not stock ShapeNet and are easy to confuse:

| id | name | note |
|---|---|---|
| `02691156` | `airplane` | the real ShapeNet airplane synset |
| `02691155` | `pasta` | a custom dataset, **not** shipped with this repo |

## 🎓 Training

### Configuration

Training is controlled via [default_config.py](default_config.py). Key configurations:

**VAE Configuration:**
```python
cfg.vae.latent_dim = 1024          # Total latent width
cfg.vae.quantizer = 'softvq'       # 'softvq' or 'kl'
cfg.vae.input_dim = 3              # Input dimension (3 for XYZ)
cfg.vae.softvq.e_dim = 8           # Width of one latent token
cfg.vae.softvq.num_codebooks = 128
```

The latent is a **sequence of tokens**, not a flat vector:
`n_tokens = latent_dim // e_dim` (1024 // 8 = 128), each of width `e_dim`.
Two constraints are asserted at construction:

- `n_tokens` must be divisible by `softvq.num_codebooks`
- `n_tokens` must not exceed the number of centers emitted by the last
  encoder SA block (128 by default)

So if you raise `latent_dim`, raise `e_dim` with it or widen the encoder's
last block.

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

`train.py` spawns its own workers with `torch.multiprocessing.spawn`, so do
**not** wrap it in `torchrun` / `torch.distributed.launch`. Just ask for the
GPUs:

```bash
python train.py --num_gpus 4
```

The provided scripts already set `NGPU` for you. The backend is NCCL.

### Checkpoints

Each run writes to `<save_dir>/checkpoints/`:

| File | What it is |
|---|---|
| `best_eval.pth` | best validation score so far — the one to sample from |
| `snapshot.pth` | latest periodic save, used for preemption recovery |
| `epoch_<n>_iters_<s>.pt` | periodic archive every `vis.save_freq` epochs |

`best_eval.pth` is selected by reconstruction quality (Chamfer) for the VAE
stage, and by held-out flow-matching loss for the prior stage — each stage
scores the thing it is actually training. Both are measured on EMA weights,
which is what sampling uses.

### Resume Training

```bash
CHECKPOINT=../experiments/vae_softvq/<exp>/checkpoints/snapshot.pth \
    bash scripts/resume.sh
```

Or manually:
```bash
python train.py --resume --pretrained <checkpoint_path>
```

This works for either stage — a prior snapshot carries its own copy of the
frozen VAE weights, so no `--vae_checkpoint` is needed when resuming.

Training also auto-resumes after preemption if `snapshot.pth` is already
present in the run directory, but only if the run has a stable name: pass
`--exp_name <name>`, otherwise each launch generates a fresh timestamped
directory and starts over.

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
- Point-voxel U-Net velocity field, conditioned on the latent tokens by
  cross-attention
- Trained by conditional flow matching; sampling integrates the ODE with
  `torchdiffeq` (`dopri5` by default)
- Generates point clouds of any requested size from latent codes

### Flow Matching Prior

- Learns prior distribution over latent codes
- Supports:
  - Optimal Transport (OT) method
  - Conditional flow matching
  - Hybrid coupling layers
  - Multi-head attention

## 📈 Evaluation

### Metrics

Generation quality is scored with the standard set-level metrics, each
computed under three distances — Chamfer (CD), Earth Mover (EMD) and F1:

| Key | Meaning | Better |
|---|---|---|
| `lgan_mmd-{CD,EMD,F1}` | Minimum Matching Distance — fidelity | lower |
| `lgan_cov-{CD,EMD,F1}` | Coverage — fraction of the reference set matched | higher |
| `1-NN-{CD,EMD,F1}-acc` | 1-NN classifier accuracy — distinguishability | 0.5 |
| `jsd` | Jensen–Shannon divergence of occupancy histograms | lower |

Reconstruction (the VAE stage's validation) reports per-shape CD, EMD and F1
directly.

EMD and Chamfer come from the JIT-compiled CUDA extensions, so evaluation
requires a GPU — there is no CPU fallback.

### Evaluation Script

```bash
CKPT=../experiments/prior/<exp>/checkpoints/best_eval.pth bash scripts/test.sh
```

Or programmatically:
```python
from utils.eval_helper import compute_score

# generated_pcs / reference_pcs: (N, Npoints, 3) tensors in world scale,
# or paths to .pt files holding them.
metrics = compute_score(generated_pcs, reference_pcs, batch_size_test=50)
# -> {'lgan_mmd-CD': ..., 'lgan_cov-CD': ..., '1-NN-CD-acc': ..., 'jsd': ...}
```

## 💻 Usage Examples

> **Latent shape.** `encode()` returns `(B, n_tokens, e_dim)` — a *sequence*,
> not a flat `(B, latent_dim)` vector. Anything you hand to `decode()` must
> have that shape, which is also what `Prior.sample()` produces.

### Generate Samples

Generation needs the prior; the VAE alone has no usable distribution over
latents. The supported entry point is [main.py](main.py), which loads the
checkpoint, applies the EMA weights and denormalises the output for you:

```bash
python main.py --pretrained ../experiments/prior/<exp>/checkpoints/best_eval.pth \
    --num_samples 64 --render --metrics
```

Programmatically:

```python
import torch
from default_config import cfg
from models import VAE, Prior

cfg.merge_from_file("../experiments/prior/<exp>/config.yml")

ckpt = torch.load("../experiments/prior/<exp>/checkpoints/best_eval.pth",
                  weights_only=False)

vae = VAE(cfg).to('cuda').eval()
vae.load_state_dict(ckpt['vae'])      # the frozen VAE, stored in the prior ckpt

prior = Prior(cfg).to('cuda').eval()
prior.load_state_dict(ckpt['model'])  # raw weights; ckpt['ema'] holds the EMA
                                      # shadow that main.py applies instead

with torch.no_grad():
    latents = prior.sample(batch_size=4)          # (4, n_tokens, e_dim)
    samples = vae.decode(latents, n_sampled_points=2048)  # (4, 2048, 3)
```

Samples come out in the dataset's normalised space; multiply by
`dataset.all_points_std` and add `all_points_mean` for world scale (or let
`main.py` do it).

### Encode Point Clouds

```python
point_clouds = torch.randn(4, 2048, 3).to('cuda')   # (B, N, 3)

latent, entropy_loss, info = vae.encode(point_clouds)
# latent: (4, cfg.vae.latent_dim // cfg.vae.softvq.e_dim, cfg.vae.softvq.e_dim)
# info:   codebook usage / temperature diagnostics
```

### Reconstruction

```python
reconstructed = vae.recont(point_clouds)   # (4, 2048, 3)
```

`recont()` is `encode` followed by `decode` and returns the reconstruction
only — no loss. Use `vae(p, g, step)` if you want the training losses.

### Interpolation

```python
clouds = vae.interpolate(pc1, pc2, n_steps=10)   # list of (1, N, 3)
```

Interpolating tokenwise in the quantised latent space is linear but not
geodesic; with `softvq.l2_norm = True` the codes lie near a sphere, so
intermediate points drift slightly off the manifold the decoder was trained
on. Treat it as a qualitative tool.

## 🔧 Troubleshooting

### CUDA Extension Compilation Errors

**Error:** `RuntimeError: Error building extension '_pvcnn_backend'` (or
`emd_ext`, `chamfer_3D`) on the first run.

Scroll up — `ninja` prints the real compiler error above the traceback.
The usual causes:

| Symptom | Fix |
|---|---|
| `nvcc: command not found` | `cuda-toolkit` missing or not on `PATH`; `conda activate delight`, check `nvcc --version` |
| `unsupported GNU version` | host `gcc` is newer than the CUDA release accepts — install an older `gcc`/`g++` and point `CUDAHOSTCXX` at it |
| `No such file or directory: 'ninja'` | `pip install ninja` |
| Stale objects after a toolchain change | `rm -rf ~/.cache/torch_extensions` and re-run |

Set `TORCH_CUDA_ARCH_LIST` to your GPU's compute capability (the provided
scripts use `7.0` for Volta) to avoid compiling for every architecture.

### Out of Memory (OOM) Errors

**Solutions:**
- Reduce batch size: `--opt data.batch_size 16`
- Reduce number of points: `--opt data.n_sample_points 1024` (must stay above
  the first encoder SA block's 1024 centers, so this is the floor)
- Narrow the decoder: `--opt vae.flow.widths "(32, 64, 128, 256)"`

Shrinking `vae.latent_dim` alone will usually trip an assertion — the token
count must stay divisible by `softvq.num_codebooks`. Lower `num_codebooks`
to match, e.g. `latent_dim 512`, `e_dim 8`, `num_codebooks 64`.

### Dataset Not Found

`ValueError: Check the data path`, or `Directory missing: ...`

**Solution:**
- Data must be at `./data/ShapeNetCore.v2.PC15k/<synsetid>/<split>/*.npy`,
  resolved relative to your working directory — launch from the repo root
- Check the synset id for your category in `synsetid_to_cate`
  ([datasets/dataset.py](datasets/dataset.py)); note `02691155` is `pasta`,
  not airplane
- All three of `train/`, `val/` and `test/` must exist for the category
- A symlink at `data/ShapeNetCore.v2.PC15k` works fine

### Poor Generation Quality

**Potential causes and fixes:**
- **Sampling from a VAE checkpoint.** `main.py` warns about this: without a
  prior, codes are drawn uniformly from the codebooks, which is a sanity
  check, not generation. Point `--pretrained` at a prior checkpoint.
- **Reconstruction is bad too** → it is the VAE, not the prior. Check the
  `codebook_usage` / `z_cos` scalars in TensorBoard; usage collapsing toward
  zero means the codebooks are dying, so lower `softvq.tau` or raise
  `softvq.entropy_loss_ratio`.
- **Reconstruction is good but samples are not** → it is the prior. Watch
  `eval/prior_fm_loss`, and train longer.
- Incorrect data normalization: check `cfg.data.normalize_global`
- Hyperparameter tuning: learning rate, KL weight, `anneal_portion`

## 📚 References

- **PointNet++**: [Original Paper](https://arxiv.org/abs/1612.00593)
- **Latent Point Diffusion Models for 3D Shape Generation**: [LION](https://arxiv.org/abs/2210.06978)
- **Vector Quantized Autoencoders**: [VQ-VAE](https://arxiv.org/abs/1711.00937)
- **SoftVQ-VAE**: [SoftVQ-VAE](https://arxiv.org/abs/2412.10958)
- **Conditional Flow Matching**: [CFM Paper](https://arxiv.org/abs/2210.02747)
- **Optimal Transport**: [OT Planning](https://arxiv.org/abs/2403.13117)
- **ShapeNet Dataset**: [Official Website](https://shapenet.org/)

## 📝 License

This project is licensed under an Academic Software License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email:** alifuatpc@gmail.com, jackhaochen@gmail.com
- **GitHub:** [alifuatsahin](https://github.com/alifuatsahin), [Jhc999](https://github.com/Jhc999)

---
