from models import VAE, Prior
from utils.ema import EMA
from utils import utils
from default_config import cfg as config
import torch
import os
from loguru import logger
import time
import matplotlib.pyplot as plt

def add_module_prefix(state_dict):
    """Add 'module.' prefix to every key in a state dict."""
    state_dict_new = {}
    for k, v in state_dict.items():
        # Avoid double prefixing if already present
        if not k.startswith('module.'):
            kn = 'module.' + k
        else:
            kn = k
        state_dict_new[kn] = v
    return state_dict_new

def filter_name(ckpt):
    ckpt_new = {}
    for k, v in ckpt.items():
        if k[:7] == 'module.':
            kn = k[7:]
        elif k[:13] == 'model.module.':
            kn = k[13:]
        elif k[:6] == 'module':
            kn = k[6:]
        else:
            kn = k
        ckpt_new[kn] = v
    return ckpt_new

def plot_pcs(pcs, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    for i, pc in enumerate(pcs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='r')
        ax.set_title(f"Sample {i+1}")
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"sample_{i+1}.png"))
        plt.close(fig)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '../experiments/prior/airplane_bs32_20250826_042427/checkpoints/snapshot.pth'
    config_path = os.path.dirname(path) + '/../config.yml'
    config.merge_from_file(config_path)
    vae = VAE(cfg=config).to(device)
    ckpt = torch.load(path)
    vae.load_state_dict(filter_name(ckpt['vae']))
    vae.eval()  # Set VAE to eval mode
    utils.requires_grad(vae, False)  # Freeze VAE weights

    prior = Prior(cfg=config).to(device)
    prior.load_state_dict(filter_name(ckpt['model']))
    if 'ema' in ckpt.keys():
        ema_prior = EMA(prior)
        ema_ckpt = filter_name(ckpt['ema'])
        ema_prior.load_state_dict(ema_ckpt)
        ema_prior.copy_to(prior)

    prior.eval()  # Set Prior to eval mode
    utils.requires_grad(prior, False)  # Freeze Prior weights

    logger.info("Starting evaluation...")
    sample_points = [2048, 8192, 15000, 100000]
    plots = []
    batch_size = 10

    # Start evaluation
    for n_sampled_points in sample_points:
        start_time = time.time()
        sample = prior.sample(batch_size=batch_size)
        output, _ = vae.decoder.decode(sample, n_sampled_points=n_sampled_points)
        elapsed_time = time.time() - start_time
        plots.append(output[-1].cpu().detach())
        logger.info(f"Evaluation completed for {n_sampled_points} points in {elapsed_time/batch_size:.2f} seconds.")

    plot_pcs(plots)
