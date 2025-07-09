from modules.layers import MLP

import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()

        self.kl_weight = cfg.model.klquantizer.kl_weight
        self.latent_dim = cfg.model.latent_dim

        self.mlp = nn.Linear(input_dim, self.latent_dim * 2)

        with torch.no_grad():
            # Initialize mu part (first half) with small std
            self.mlp.weight.data[:self.latent_dim].normal_(std=0.0033)
            self.mlp.bias.data[:self.latent_dim].fill_(0.0)
            
            # Initialize logvar part (second half) with different std
            self.mlp.weight.data[self.latent_dim:].normal_(std=0.033)
            self.mlp.bias.data[self.latent_dim:].fill_(0.0)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + eps * std
    
    def forward(self, features):
        """
        Forward pass for the KLQuantizer module.
        """
        # get posterior distribution from point cloud features
        features = self.mlp(features)
        mus, log_vars = features[:, :self.latent_dim], features[:, self.latent_dim:]
        g_samples = self.reparameterize(mus, log_vars)

        kl_loss = self.compute_kl_loss(mus, log_vars)

        return g_samples, kl_loss, (mus, log_vars)

    def compute_kl_loss(self, posterior_mus, posterior_logvars):
        """
        Compute the KL divergence loss for the quantizer.
        Args:
            posterior_mus: Mean of the posterior distribution.
            posterior_logvars: Log variance of the posterior distribution.
        Returns:
            kl_loss: Computed KL divergence loss.
        """
        kl_loss = -0.5 * torch.sum(
            1 + posterior_logvars 
            - posterior_mus.pow(2) 
            - posterior_logvars.exp()
        )
        return kl_loss * self.kl_weight
    
    @torch.no_grad()
    def sample(self, batch_size, device=None):
        """
        Sample random codes from the codebook.
        Args:
            batch_size: Number of samples to generate.
            device: Device to place the samples on.
        Returns:
            z_random: Randomly sampled codes.
            indices: Indices of the sampled codes.
        """
        sample = torch.randn(batch_size, self.latent_dim, device=device)

        return sample

    @property
    def device(self):
        """Get the device of the model parameters"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Fallback if no parameters
            return torch.device('cpu')