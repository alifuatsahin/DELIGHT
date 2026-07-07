import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq  # For ODE integration
from typing import Tuple, Dict

from modules.cfm import get_CFM
from modules.flows import PriorFlow

class Prior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.prior.depth
        self.width = cfg.prior.width
        self.e_dim = cfg.vae.softvq.e_dim
        self.seq_len = cfg.vae.latent_dim // self.e_dim
        self.solver = cfg.prior.solver
        self.atol = cfg.prior.atol
        self.rtol = cfg.prior.rtol

        self.model = PriorFlow(cfg)
        self.FM = get_CFM(cfg.prior)

        self.normalize_latents = getattr(cfg.prior, 'normalize_latents', True)
        self.stats_samples = getattr(cfg.prior, 'latent_stats_samples', 100000)
        self.register_buffer('latent_sum', torch.zeros(self.e_dim))
        self.register_buffer('latent_sumsq', torch.zeros(self.e_dim))
        self.register_buffer('latent_count', torch.zeros(1))
        self.register_buffer('latent_mean', torch.zeros(1, 1, self.e_dim))
        self.register_buffer('latent_std', torch.ones(1, 1, self.e_dim))
        self._stats_done = False

    @torch.no_grad()
    def _update_latent_stats(self, latents: torch.Tensor) -> None:
        """Accumulate per-channel latent statistics until the warmup budget."""
        if not self.normalize_latents or self._stats_done:
            return
        if self.latent_count.item() >= self.stats_samples:
            # Cached so the warmup check stops forcing a device sync every step.
            self._stats_done = True
            return
        flat = latents.detach().reshape(-1, self.e_dim).float()
        self.latent_sum += flat.sum(dim=0)
        self.latent_sumsq += (flat ** 2).sum(dim=0)
        self.latent_count += flat.shape[0]

        n = self.latent_count.clamp_min(1.0)
        mean = self.latent_sum / n
        var = (self.latent_sumsq / n - mean ** 2).clamp_min(1e-8)
        self.latent_mean.copy_(mean.view(1, 1, -1))
        self.latent_std.copy_(var.sqrt().view(1, 1, -1))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latents:
            return latents
        return (latents - self.latent_mean) / self.latent_std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latents:
            return latents
        return latents * self.latent_std + self.latent_mean

    def sample(
        self,
        batch_size: int,
        num_steps: int = 100,
    ) -> torch.Tensor:

        x0 = torch.randn(batch_size, self.seq_len, self.e_dim, device=self.device)

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, t.expand(batch_size).unsqueeze(-1)),
            x0,
            torch.linspace(0, 1, num_steps, device=x0.device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )

        # The flow is trained in standardised latent space; map back to the
        # space the VAE decoder expects.
        return self.denormalize(traj[-1])

    def forward(
        self,
        latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:

        if self.training:
            self._update_latent_stats(latents)
        latents = self.normalize(latents)

        x0 = torch.randn_like(latents)

        t = torch.rand(x0.shape[0], device=x0.device).unsqueeze(-1)#.expand(-1, self.seq_len)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, latents, t)

        vt = self.model(xt, t)
        loss = torch.mean((vt - ut) ** 2)

        logs = {
            'latent_std': self.latent_std.mean(),
            'latent_mean_abs': self.latent_mean.abs().mean(),
        }

        return loss, logs
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device