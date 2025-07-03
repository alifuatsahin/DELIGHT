from model.vae import VAE
from default_config import cfg

import torch


if __name__ == "__main__":
    model = VAE(cfg, mode='training')
    model.eval()  # Set the model to evaluation mode

    B = 2
    input_dim = cfg.input_dim
    N = 2048  # Number of points, can be adjusted as needed

    p = torch.randn(B, input_dim, N)  # e.g., B=2, input_dim=3, N=2048
    g = torch.randn(B, input_dim, N)  # or whatever your encoder expects
    out = model(p, g)