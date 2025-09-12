import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq  # For ODE integration
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict

from modules.flows import FlowBase, ExpBase, PVCNN2Unet, Exp2Base, Exp3Base, Exp4Base, Exp5Base
from modules.layers import MLPGaussian, StandartGaussian
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

        self.model = Exp5Base(cfg)
        # SA_BLOCKS = [ # conv_configs, sa_configs
        # ((32, 2, 32), (1024, 0.1, 32, (16, 32))),
        # ((64, 1, 16), (256, 0.2, 32, (32, 64))),
        # ((128, 1, 8), (64, 0.4, 32, (64, 128))),
        # # (None, (16, 0.8, 32, (128, 128, 128))),
        # ]
        # FP_BLOCKS = [
        #     ((128, 128), (128, 1, 8)), # fp_configs, conv_configs
        #     # ((128, 128), (128, 1, 8)),
        #     ((128, 128), (64, 1, 16)),
        #     ((128, 128, 64), (32, 2, 32)),
        # ]
        # self.model = PVCNN2Unet(emb_dim=cfg.flow.t_emb_ch,
        #                         context_dim=cfg.softvq.e_dim, 
        #                         input_dim=self.input_dim,
        #                         extra_feature_channels=0, 
        #                         sa_blocks=SA_BLOCKS, 
        #                         fp_blocks=FP_BLOCKS)
        
        self.FM = get_CFM(cfg.flow)

        if cfg.point_prior_n_layers > 0:
            # Prior network for initial point generation
            self.point_prior = MLPGaussian(
                n_layers=cfg.point_prior_n_layers,
                in_features=self.latent_dim,
                out_features=cfg.input_dim,
                mu_weight_std=0.001,
                mu_bias=0.0,
                deterministic=False,
                logvar_weight_std=0.01,
                logvar_bias=0.0
            )
        else:
            self.point_prior = StandartGaussian(
                out_features=cfg.input_dim
            )
        
    def _validate_config(self):
        """Validate decoder configuration parameters."""
        assert self.n_flows > 0, "Number of flows must be positive"
        assert self.depth > 0, "Depth must be positive"
        assert self.width > 0, "Feature dimension must be positive"
        assert self.latent_dim > 0, "Latent dimension must be positive"
        assert self.input_dim > 0, "Input dimension must be positive"

    def get_weights(
        self, 
        latent_vector: torch.Tensor, 
        warmup: bool = False
    ) -> torch.Tensor:
        """
        Get the mixture weights for the decoder flows.
        Args:
            latent_vector: latent features (B, latent_dim)
            warmup: whether to use fixed uniform weights
            
        Returns:
            mixture_weights: weights for each flow (B, n_flows)
        """
        if warmup:
            batch_size = latent_vector.shape[0]
            log_weights = torch.log(torch.full((batch_size, self.n_flows), 1.0 / self.n_flows, device=latent_vector.device))
            return log_weights
        else:
            # Use learned weights based on latent vector
            log_weights = self.mixture_weights_enc(latent_vector)
            return log_weights

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = latents.shape[0]

        p_prior_mus, p_prior_logvars = self.point_prior(latents.view(B, -1))

        # Expand prior parameters to match point dimensions
        p_prior_mus = p_prior_mus.unsqueeze(1).expand(B, n_sampled_points, self.input_dim)
        p_prior_logvars = p_prior_logvars.unsqueeze(1).expand(B, n_sampled_points, self.input_dim)

        x0 = self.reparametrize(p_prior_mus, p_prior_logvars)  # Initial point cloud

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, t.expand(x.shape[0], 1), context=latents),
            x0,
            torch.linspace(0, 1, num_steps, device=x0.device),
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver,
        )

        return traj[-1], torch.ones(latents.shape[0], n_sampled_points, device=latents.device)

    def forward(
        self, 
        p: torch.Tensor, 
        latents: torch.Tensor, 
        warmup: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, C = p.shape

        p_prior_mus, p_prior_logvars = self.point_prior(latents.view(latents.shape[0], -1), warmup=warmup)

        # Expand prior parameters to match point dimensions
        p_prior_mus = p_prior_mus.unsqueeze(1).expand(B, N, C)
        p_prior_logvars = p_prior_logvars.unsqueeze(1).expand(B, N, C)

        x0 = self.reparametrize(p_prior_mus, p_prior_logvars)  # Initial point cloud

        t = torch.rand(x0.shape[0], device=x0.device).unsqueeze(-1)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, p, t)

        vt = self.model(xt, t, context=latents)
        loss = torch.mean((vt - ut) ** 2)

        return loss, [1]  # Placeholder for flow labels, replace with actual labels if needed
    
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