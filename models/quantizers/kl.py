import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()

        self.latent_dim = cfg.latent_dim
        self.e_dim = cfg.softvq.e_dim
        assert self.latent_dim % self.e_dim == 0, \
            f"latent_dim ({self.latent_dim}) must be divisible by e_dim ({self.e_dim})"
        self.sequence_len = self.latent_dim // self.e_dim
        self.pre_quant_layer = nn.Linear(input_dim, self.latent_dim * 2)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + eps * std
    
    @staticmethod
    def sample_gaussian(size, device):
        y = torch.randn(*size).float().to(device)

        return y
    
    def forward(self, features, *args, **kwargs):
        """
        Forward pass for the KLQuantizer module.
        """
        # Global max pooling to get fixed-size representation
        # features: (B, D, N) -> (B, D)
        features = features.max(dim=-1)[0]

        # get posterior distribution from point cloud features
        features = self.pre_quant_layer(features)
        mus, log_vars = features[:, :self.latent_dim], features[:, self.latent_dim:]
        g_samples = self.reparameterize(mus, log_vars)

        kl_loss = self.compute_kl_loss(mus, log_vars)

        info = {
            "mean_mus": torch.mean(mus, dim=0).mean(),
            "mean_logvars": torch.mean(log_vars, dim=0).mean(),
        }

        # (B, latent_dim) -> (B, sequence_len, e_dim)
        g_samples = g_samples.view(-1, self.sequence_len, self.e_dim)

        return g_samples, kl_loss, info

    def compute_kl_loss(self, mus, logvars):
        """
        Compute the KL divergence loss for the quantizer.
        Args:
            mus: Mean of the posterior distribution.
            logvars: Log variance of the posterior distribution.
        Returns:
            kl_loss: Computed KL divergence loss.
        """
        kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=1).mean()
        return kl_loss / 2e4
    
    @torch.no_grad()
    def sample(self, batch_size, device=None):
        """
        Sample random codes from the standard gaussian prior.

        Args:
            batch_size: Number of samples to generate.
            device: Device to place the samples on.
        Returns:
            z_random: (batch_size, sequence_len, e_dim) latent tokens, matching
                the shape returned by forward().
        """
        if device is None:
            device = self.pre_quant_layer.weight.device
        sample = self.sample_gaussian(
            (batch_size, self.sequence_len, self.e_dim), device=device)

        return sample