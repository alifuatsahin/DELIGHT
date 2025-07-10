from modules.flow import get_point_cnf
from modules.layers import MLP
from utils.eval_helper import standard_normal_logprob

import torch

class Decoder:
    def __init__(self, cfg):
        self.model = get_point_cnf(cfg)
        self.input_dim = cfg.input_dim

        self.point_prior = MLP(
            n_layers=cfg.point_prior_n_layers,
            in_features=cfg.latent_dim,
            out_features=cfg.input_dim,
            mu_weight_std=0.001,
            mu_bias=0.0,
            deterministic=False,
            logvar_weight_std=0.01,
            logvar_bias=0.0
        )

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def forward(self, p, g, ):
        num_points = p.shape[2]
        batch_size = p.shape[0]

        y, delta_log_py = self.model(p, g, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py
        recon_loss = -log_px.mean()

        return recon_loss
    
    def decode(self, z, num_points=2048, truncate_std=None):
        mus, logvars = self.point_prior(z)

        mus = mus.unsqueeze(2).expand(-1, self.input_dim, num_points)
        logvars = logvars.unsqueeze(2).expand(-1, self.input_dim, num_points)

        y = self.reparameterize_gaussian(mus, logvars)
        x = self.model(y, z, reverse=True).view(*y.size())

        return x
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device