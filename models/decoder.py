import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq  # For ODE integration
from typing import List, Tuple, Optional, Dict, Any

from modules.flows import UNetFlow, UNetFlow2, UNetFlow3
from modules.cfm import get_CFM

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_flows = cfg.flow.n_flows
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.latent_dim = cfg.latent_dim
        self.input_dim = cfg.input_dim
        self.solver = cfg.flow.solver
        self.atol = cfg.flow.atol
        self.rtol = cfg.flow.rtol
        
        # Validate configuration
        self._validate_config()

        self.model = UNetFlow(
            depth=cfg.flow.depth,
            widths=cfg.flow.widths,
            radiuses=cfg.flow.radiuses,
            input_dim=cfg.input_dim,
            e_dim=cfg.softvq.e_dim,
            t_emb_ch=cfg.flow.t_emb_ch,
            patch_size=cfg.flow.patch_size,
            num_centers=cfg.flow.num_centers,
            n_heads=cfg.flow.n_heads,
            num_neighbors=cfg.flow.num_neighbors,
            use_xformers=cfg.flow.use_xformers,
        )

        self.FM = get_CFM(cfg.flow)

    def _validate_config(self):
        """Validate decoder configuration parameters."""
        assert self.n_flows > 0, "Number of flows must be positive"
        assert self.depth > 0, "Depth must be positive"
        assert self.width > 0, "Feature dimension must be positive"
        assert self.latent_dim > 0, "Latent dimension must be positive"
        assert self.input_dim > 0, "Input dimension must be positive"

    def reparametrize(
        self, 
        mus: torch.Tensor, 
        logvars: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent space.
        Args:
            mus: means of the distributions (B, D, N)
            logvars: log variances of the distributions (B, D, N)
            
        Returns:
            samples: sampled features (B, D, N)
        """
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mus)
    
    def decode(
        self, 
        latents: torch.Tensor, 
        n_sampled_points: int,
        num_steps: int = 100,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = latents.shape[0]

        x0 = torch.randn(B, n_sampled_points, self.input_dim, device=latents.device).float()  # Initial point cloud

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, t.expand(x.shape[0]), context=latents),
            x0,
            torch.linspace(0, 1, num_steps, device=x0.device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )
        if return_trajectory:
            return traj
        return traj[-1]

    def forward(
        self, 
        p: torch.Tensor, 
        latents: torch.Tensor, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, C = p.shape

        x0 = torch.randn(B, N, C, device=p.device).float()  # Initial point cloud

        t = torch.rand(x0.shape[0], device=x0.device)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, p, t)

        vt = self.model(xt, t, context=latents)
        loss = torch.mean((vt - ut) ** 2)

        return loss
    
    def estimate_parameters(self) -> Dict[str, int]:
        """
        Estimate parameter counts for different components.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        prior_params = sum(p.numel() for p in self.point_prior.parameters())
        
        return {
            "total": total_params,
            "decoder_flows": decoder_params,
            "point_prior": prior_params,
        }
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        device = self.get_device()
        if device.type != 'cuda':
            return {"error": "Model not on CUDA device"}
            
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        cached = torch.cuda.memory_reserved(device) / 1024**2
        
        return {
            "allocated_mb": allocated,
            "cached_mb": cached,
            "free_mb": cached - allocated
        }