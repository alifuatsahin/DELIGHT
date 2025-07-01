import torch.nn as nn
from utils.checker import *
from .dense import dense

class AdaGN(nn.Module):
    def __init__(self, ndim, cfg, n_channel):
        super.__init__()
        style_dim = cfg.latent_pts.style_dim
        init_scale = cfg.latent_pts.init_scale
        self.ndim = ndim
        self.n_channel = n_channel
        self.style_dim = style_dim
        