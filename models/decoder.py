from modules.flow import get_point_cnf
from utils.eval_helper import standard_normal_logprob
from modules.fre_loss import fre_loss

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = get_point_cnf(cfg)
        self.input_dim = cfg.input_dim
        self.high_freq_recon_coeff = cfg.high_freq_recon_coeff
        self.high_freq_recon_lmax = cfg.high_freq_recon_lmax

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps
    
    @staticmethod
    def sample_gaussian(size, device):
        """Sample from a standard normal distribution."""
        y = torch.randn(*size).float().to(device)
        return y

    def forward(self, p, g, num_points=None):
        if num_points is None:
            num_points = p.shape[1]
        batch_size = p.shape[0]

        y, delta_log_py = self.model(p, g, torch.zeros(batch_size, num_points, 1).to(p))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py
        recon_loss = -log_px.mean()

        if self.high_freq_recon_coeff > 0:
            fre_loss_item = fre_loss(p, y, lmax=self.high_freq_recon_lmax) * 10 ** 7
            recon_loss = (1 - self.high_freq_recon_coeff) * recon_loss + self.high_freq_recon_coeff * fre_loss_item

        return recon_loss
    
    def decode(self, z, num_points=2048):
        y = self.sample_gaussian((z.shape[0], num_points, self.input_dim), device=self.device)
        x = self.model(y, z, reverse=True).view(*y.size())

        return x
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device