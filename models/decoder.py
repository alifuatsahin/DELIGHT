import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq  # For ODE integration
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict

from modules.flows import FlowAttn, FlowBase
from modules.layers import MLP, StandartGaussian
from modules.cfm import get_CFM
    
class WeightsMLP(nn.Module):
    def __init__(
        self,
        n_layers,
        in_features,
        out_features,
        out_weight_std=0.001,
        out_bias=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.out_features = out_features
        self.out_weight_std = out_weight_std
        self.out_bias = out_bias

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_swish'.format(i), nn.SiLU())

        self.output = nn.Sequential(OrderedDict([
            ('out_mlp0', nn.Linear(in_features, out_features, bias=True)),
        ]))

        with torch.no_grad():
            self.output[0].weight.data.normal_(std=out_weight_std)
            nn.init.constant_(self.output[0].bias.data, out_bias)

    def forward(self, input):
        feats = self.features(input)
        output = self.output(feats)
        log_weights = F.log_softmax(output, dim=-1)

        return log_weights

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base = cfg.flow.base
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

        if self.base == 'attn':
            self.model = FlowAttn(cfg)
        elif self.base == 'resnet':
            self.model = FlowBase(cfg)
        else:
            raise ValueError(f"Unsupported flow base type: {self.base}")
        self.FM = get_CFM(cfg.flow)

        if cfg.point_prior_n_layers > 0:
            # Prior network for initial point generation
            self.point_prior = MLP(
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
        p_prior_mus = p_prior_mus.unsqueeze(2).expand(B, self.input_dim, n_sampled_points)
        p_prior_logvars = p_prior_logvars.unsqueeze(2).expand(B, self.input_dim, n_sampled_points)

        x0 = self.reparametrize(p_prior_mus, p_prior_logvars)  # Initial point cloud

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward(x, t.expand(x.shape[0], n_sampled_points), context=latents),
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

        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, p)

        xt = xt.transpose(1, 2).contiguous()  # (B, C, N)
        ut = ut.transpose(1, 2).contiguous()  # (B, C, N)
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