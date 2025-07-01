from .encoder import Encoder
from .decoder import Decoder

import torch.nn as nn

class VAE(nn.Module):
    def __init__(
        self,
        n_flows,
        depth,
        feat_dim,
        latent_dim,
        weight_std=0.01
    ):
        super().__init__()
        self.n_flows = n_flows
        self.depth = depth
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.weight_std = weight_std

        self.encoder = Encoder(depth, feat_dim, latent_dim, weight_std=weight_std)
        self.decoder = Decoder(n_flows, depth, feat_dim, latent_dim, weight_std=weight_std)

    def forward(self, p, g, mode='direct'):
        z_mean, z_logvar = self.encoder(p, g)
        z = self.reparameterize(z_mean, z_logvar)
        
        if mode == 'direct':
            return self.decoder(z, g, mode=mode), z_mean, z_logvar
        elif mode == 'inverse':
            return self.decoder(z, g, mode=mode), z_mean, z_logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std