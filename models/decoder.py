import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict

from modules.flows import CondRealNVPFlow3DTriple
from modules.layers import MLP
from modules.fre_loss import fre_loss
    
class WeightsMLP(nn.Module):
    def __init__(
        self,
        n_layers,
        in_features,
        out_features,
        alpha_weight_std=0.001,
        alpha_bias=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_weight_std = alpha_weight_std
        self.alpha_bias = alpha_bias

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                self.features.add_module('mlp{}_bn'.format(i), nn.LayerNorm(in_features))
                self.features.add_module('mlp{}_swish'.format(i), nn.GELU())

        self.alphas = nn.Sequential(OrderedDict([
            ('alpha_mlp0', nn.Linear(in_features, out_features, bias=True)),
            ('softplus', nn.Softplus())
        ]))

        with torch.no_grad():
            self.alphas[0].weight.data.normal_(std=alpha_weight_std)
            nn.init.constant_(self.alphas[0].bias.data, alpha_bias)

    def forward(self, input):
        if self.n_layers > 0:
            features = self.features(input)
        else:
            features = input
        alphas = torch.clamp(self.alphas(features) + 1, min=1)
        dirichlet = torch.distributions.Dirichlet(alphas)
        weights = dirichlet.rsample()

        return weights

class DecBlock(nn.Module):
    def __init__(
        self,
        depth: int,
        feat_dim: int,
        latent_dim: int,
        weight_std: float = 0.01
    ):
        super().__init__()
        self.depth = depth
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.weight_std = weight_std

        # Create flow layers with alternating patterns
        self.layers = nn.ModuleList([
            CondRealNVPFlow3DTriple(
                feat_dim, latent_dim,
                weight_std=self.weight_std, 
                pattern=(i % 2)
            ) for i in range(self.depth)
        ])
        
    def forward(
        self, 
        p: torch.Tensor, 
        g: torch.Tensor, 
        mode: str = 'direct'
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:

        ps, mus, logvars = [], [], []
        
        for i in range(self.depth):
            if mode == 'direct':
                cur_p = p if i == 0 else ps[-1]
                buf = self.layers[i](cur_p, g, mode=mode)
                ps = ps + buf[0]
                mus = mus + buf[1]
                logvars = logvars + buf[2]
            elif mode == 'inverse':
                cur_p = p if i == 0 else ps[0]
                buf = self.layers[-(i + 1)](cur_p, g, mode=mode)
                ps = buf[0] + ps
                mus = buf[1] + mus
                logvars = buf[2] + logvars

        return ps, mus, logvars

    
class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_flows = cfg.n_flows
        self.depth = cfg.depth
        self.feat_dim = cfg.feat_dim
        self.latent_dim = cfg.latent_dim
        self.input_dim = cfg.input_dim
        self.high_freq_recon_coeff = cfg.high_freq_recon_coeff
        self.high_freq_recon_lmax = cfg.high_freq_recon_lmax
        
        # Validate configuration
        self._validate_config()

        # Compute optimal parameters under budget
        self.flow_depth, self.feat_dim = self._get_decoder_params(min_feat_dim=4)

        # Create decoder flows
        self.decoder = nn.ModuleList([
            DecBlock(
                self.flow_depth,
                self.feat_dim,
                self.latent_dim,
                weight_std=0.01
            ) for _ in range(self.n_flows)
        ])
        
        # Network to compute mixture weights from latent codes
        self.mixture_weights_enc = WeightsMLP(
            n_layers=cfg.weight_n_layers,
            in_features=self.latent_dim,
            out_features=self.n_flows,
            alpha_weight_std=0.001,
            alpha_bias=0.0
        )

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
        
        # Initialize parameters
        self._initialize_parameters()
        
    def _validate_config(self):
        """Validate decoder configuration parameters."""
        assert self.n_flows > 0, "Number of flows must be positive"
        assert self.depth > 0, "Depth must be positive"
        assert self.feat_dim > 0, "Feature dimension must be positive"
        assert self.latent_dim > 0, "Latent dimension must be positive"
        assert self.input_dim > 0, "Input dimension must be positive"
        
    def _initialize_parameters(self):
        """Initialize decoder parameters."""
        
        # Initialize decoder blocks
        for decoder_block in self.decoder:
            for layer in decoder_block.layers:
                if hasattr(layer, 'weight') and layer.weight is not None:
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_decoder_params(self, min_feat_dim: int = 4) -> Tuple[int, int]:
        """
        Decide feature size and number of coupling layers under a parameter budget.
        Returns:
            Tuple of (flow_depth, feat_dim)
        """
        if self.n_flows == 1:
            return self.depth, self.feat_dim

        # Compute optimal flow depth based on number of flows
        flow_depth = max(1, math.ceil(self.depth / math.sqrt(self.n_flows)))
        feat_dim = self.feat_dim
        
        # Get baseline parameter count for comparison
        baseline_count = self.get_param_count_for(self.depth, self.feat_dim)
        
        # Adjust feat_dim to stay within parameter budget
        current_count = self.get_param_count_for(flow_depth, feat_dim)
        
        while current_count > baseline_count and feat_dim > min_feat_dim:
            feat_dim -= 1
            current_count = self.get_param_count_for(flow_depth, feat_dim)
            
        # Ensure we don't go below minimum
        feat_dim = max(feat_dim, min_feat_dim)

        return flow_depth, feat_dim

    def get_param_count_for(self, flow_depth: int, feat_dim: int) -> int:
        """
        Estimate parameter count for given configuration.
        
        Args:
            flow_depth: number of flow layers
            feat_dim: feature dimension
            
        Returns:
            Estimated parameter count
        """
        # Rough estimation of CondRealNVPFlow3D parameters
        count_single_flow = 18 * feat_dim + 4 * feat_dim * self.latent_dim + 6 * feat_dim**2
        count_triple_flow = 3 * count_single_flow
        total_count = flow_depth * count_triple_flow * self.n_flows
        return total_count

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
            weights = torch.full((batch_size, self.n_flows), 1.0 / self.n_flows, device=latent_vector.device)
            return weights
        else:
            # Use learned weights based on latent vector
            weights = self.mixture_weights_enc(latent_vector)
            return weights

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
        n_sampled_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate point clouds from latent codes using mixture of flows.
        
        Args:
            latents: latent codes (B, latent_dim)
            n_sampled_points: number of points to generate per sample
            
        Returns:
            Tuple of (samples, labels, mixture_weights_logits)
            - samples: generated points (B, input_dim, n_sampled_points)
            - labels: flow assignment labels (B, n_sampled_points)  
            - mixture_weights_logits: mixture weights (B, n_flows)
        """
        assert len(latents.shape) == 2, f"Latents should be (B, D), got {latents.shape}"
        batch_size = latents.shape[0]
        device = latents.device

        # Get mixture weights for all samples at once
        mixture_weights = self.get_weights(latents, warmup=False)
        
        # Convert to probabilities
        mixture_probs = mixture_weights.cpu().numpy()

        # Pre-allocate output tensors
        all_samples = torch.zeros(
            batch_size, self.input_dim, n_sampled_points, 
            device=device, dtype=latents.dtype
        )
        all_labels = torch.zeros(
            batch_size, n_sampled_points, 
            device=device, dtype=torch.long
        )

        # Process each sample in the batch
        for b in range(batch_size):
            g = latents[b:b+1]  # Keep batch dimension
            probs = mixture_probs[b]
            
            # Sample flow assignments for this sample
            flow_assignments = np.random.choice(
                self.n_flows, 
                size=n_sampled_points, 
                p=probs
            )
            
            # Count points per flow
            flow_counts = np.bincount(flow_assignments, minlength=self.n_flows)
            
            # Generate points for each flow
            sample_parts = []
            
            for flow_idx, count in enumerate(flow_counts):
                if count == 0:
                    continue
                    
                # Generate prior samples for this flow
                p_prior_mus, p_prior_logvars = self.point_prior(g)
                
                # Expand to required number of points
                p_prior_mus = p_prior_mus.unsqueeze(2).expand(-1, self.input_dim, count)
                p_prior_logvars = p_prior_logvars.unsqueeze(2).expand(-1, self.input_dim, count)
                
                # Sample from prior
                p_prior_samples = self.reparametrize(p_prior_mus, p_prior_logvars)
                
                # Apply flow transformation
                flow_outputs = self.decoder[flow_idx](p_prior_samples, g, mode='direct')
                final_points = flow_outputs[0][-1]  # Last transformation output
                
                # Store results
                sample_parts.append((final_points, flow_idx, count))
                
            # Combine results maintaining original order
            for final_points, flow_idx, count in sample_parts:
                # Find positions for this flow in the original assignment
                flow_mask = flow_assignments == flow_idx
                flow_positions = np.where(flow_mask)[0]
                
                # Assign points and labels
                all_samples[b, :, flow_positions] = final_points[0, :, :count]
                all_labels[b, flow_positions] = flow_idx + 1  # 1-indexed labels

        return all_samples, all_labels

    def forward(
        self, 
        p: torch.Tensor, 
        g: torch.Tensor, 
        n_sampled_points: int, 
        warmup: bool = False
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """
        Forward pass during training (inverse flow to compute likelihoods).
        
        Args:
            p: input point clouds (B, input_dim, N)
            g: conditioning latent vectors (B, latent_dim)
            n_sampled_points: number of points per flow (for compatibility)
            warmup: whether to use uniform mixture weights
            
        Returns:
            Tuple of (flow_outputs, mixture_weights_logits)
            - flow_outputs: list of dicts with flow statistics
            - mixture_weights_logits: mixture weights (B, n_flows)
        """
        batch_size = g.shape[0]
        mixture_weights = self.get_weights(g, warmup=warmup)

        if self.high_freq_recon_coeff > 0:
            flow_assignments = torch.multinomial(mixture_weights, n_sampled_points, replacement=True)
            out_shape = torch.zeros(batch_size, self.input_dim, n_sampled_points, device=p.device)

        # Each flow processes the same number of points during training
        n_sample_flow = [n_sampled_points] * self.n_flows
            
        output = []

        for i, decoder_block in enumerate(self.decoder):
            # Generate output parts for each flow decoder
            flow_out = {}
            
            # Get prior parameters
            p_prior_mus, p_prior_logvars = self.point_prior(g)
            
            # Expand prior parameters to match point dimensions
            flow_out['p_prior_mus'] = [
                p_prior_mus.unsqueeze(2).expand(
                    batch_size, self.input_dim, n_sample_flow[i]
                )
            ]
            flow_out['p_prior_logvars'] = [
                p_prior_logvars.unsqueeze(2).expand(
                    batch_size, self.input_dim, n_sample_flow[i]
                )
            ]
            
            # Apply inverse flow transformation to input points
            buf = decoder_block(p, g, mode='inverse')
            
            # Combine results
            flow_out['p_prior_samples'] = buf[0] + [p]
            flow_out['p_prior_mus'].extend(buf[1])
            flow_out['p_prior_logvars'].extend(buf[2])

            if self.high_freq_recon_coeff > 0:
                for b in range(batch_size):
                    mask = (flow_assignments[b] == i)
                    out_shape[b, :, mask] = flow_out['p_prior_samples'][0][b, :, mask]
                
            output.append(flow_out)

        recon_loss = self.get_pnll(output, mixture_weights)

        if self.high_freq_recon_coeff > 0:
            fre_loss_item = fre_loss(p, out_shape, lmax=self.high_freq_recon_lmax) * 10 ** 7
            recon_loss = (1 - self.high_freq_recon_coeff) * recon_loss + self.high_freq_recon_coeff * fre_loss_item

        return recon_loss, torch.mean(mixture_weights, dim=0)

    def get_pnll(self, output, mixture_weights):
        log_weights = torch.log(mixture_weights)  # Avoid log(0)
        log_weights = log_weights.unsqueeze(1)
        
        num_patches = len(output)
        num_batches = output[0]['p_prior_mus'][0].shape[0]
        pnll = []
        for i in range(num_batches):
            loss_pnll_over_patches = []
            for j in range(num_patches):
                cur_mus = output[j]['p_prior_mus'][0][i, :, :]
                cur_logvars = output[j]['p_prior_logvars'][0][i, :, :]
                # compute sum of log determinant of each shape
                cur_log_determinant = sum(output[j]['p_prior_logvars'])[i, :, :]
                cur_samples = output[j]['p_prior_samples'][0][i, :, :]

                # compute the log probability of each shape in each flow
                part_1 = -torch.sum(cur_log_determinant + ((cur_samples - cur_mus) ** 2 / torch.exp(cur_logvars)),
                                    dim=0, keepdim=True)
                part_2 = -np.log(2.0 * np.pi) * cur_samples.shape[0]
                cur_pnll = 0.5 * torch.add(part_1, part_2)
                loss_pnll_over_patches.append(cur_pnll)
            loss_pnll_over_patches = torch.transpose(torch.cat(loss_pnll_over_patches, dim=0), 0, 1)

            # compute the log probability of each shape in all flows by adding its log weights
            log_probs_pnll = loss_pnll_over_patches + log_weights[i]
            log_probs_pnll = torch.logsumexp(log_probs_pnll, dim=-1)
            log_probs_pnll = -torch.sum(log_probs_pnll)

            pnll.append(log_probs_pnll.unsqueeze(0))

        # compute the average loss over the batch
        pnll = torch.cat(pnll)
        pnll = torch.mean(pnll)

        return pnll
    
    def estimate_parameters(self) -> Dict[str, int]:
        """
        Estimate parameter counts for different components.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        decoder_params = sum(p.numel() for decoder in self.decoder for p in decoder.parameters())
        weights_params = sum(p.numel() for p in self.mixture_weights_enc.parameters())
        prior_params = sum(p.numel() for p in self.point_prior.parameters())
        
        return {
            "total": total_params,
            "decoder_flows": decoder_params,
            "mixture_weights": weights_params,
            "point_prior": prior_params,
            "other": total_params - decoder_params - weights_params - prior_params
        }
    
    def __repr__(self) -> str:
        """String representation of the decoder."""
        param_stats = self.estimate_parameters()
        return (f"Decoder(n_flows={self.n_flows}, "
                f"flow_depth={self.flow_depth}, "
                f"feat_dim={self.feat_dim}, "
                f"latent_dim={self.latent_dim}, "
                f"total_params={param_stats['total']:,})")
    
    def get_device(self) -> torch.device:
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