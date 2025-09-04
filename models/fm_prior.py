import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq  # For ODE integration
from typing import Tuple, Dict

from modules.cfm import get_CFM
from modules.flows import PriorBase

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

        self.model = PriorBase(cfg)
        self.FM = get_CFM(cfg.prior)

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

        return traj[-1]

    def forward(
        self, 
        latents: torch.Tensor, 
    ) -> Tuple[torch.Tensor, Dict]:

        B, D, N = latents.shape

        latents = latents.transpose(1, 2).contiguous()
        x0 = torch.randn(B, N, D, device=latents.device)

        t = torch.rand(x0.shape[0], device=x0.device).unsqueeze(-1)#.expand(-1, self.seq_len)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, latents, t)

        vt = self.model(xt, t)
        loss = torch.mean((vt - ut) ** 2)

        return loss, {}
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device