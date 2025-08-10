import torch
import torch.nn as nn

class Quantizer(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()

        self.latent_dim = cfg.latent_dim
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

        return g_samples.unsqueeze(-1), kl_loss, info

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
        return kl_loss
    
    @torch.no_grad()
    def sample(self, batch_size, device):
        """
        Sample random codes from gaussian distribution.
        Args:
            batch_size: Number of samples to generate.
            device: Device to place the samples on.
        Returns:
            z_random: Randomly sampled codes.
            indices: Indices of the sampled codes.
        """
        sample = self.sample_gaussian((batch_size, self.latent_dim), device=device)

        return sample