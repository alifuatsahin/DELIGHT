from .encoder import Encoder
from .decoder import Decoder
from .quantizers import get_quantizer

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.latent_dim = cfg.model.latent_dim
        self.cfg = cfg  # Store config for debugging

        self.encoder = Encoder(cfg.model.input_dim)
        self.decoder = Decoder(cfg.model)

        self.quantizer = get_quantizer(cfg, self.encoder.out_features)

    def encode(self, x):
        """
        Args:
            x: input point clouds (B, N, 3)
        Returns:
            latent: sampled latent representation (B, latent_dim)
            kl_loss: KL divergence loss for the quantizer
        """
        encoded_features = self.encoder(x)

        latent, kl_loss = self.quantizer(encoded_features)

        return latent, kl_loss
    
    @torch.no_grad()
    def recont(self, pc, deterministic=True):
        """
        Reconstruct point clouds (encode then decode).
        
        Args:
            pc: input point clouds (B, 3, N)
            deterministic: whether to use deterministic reconstruction
            
        Returns:
            Tuple of (reconstructed_samples, flow_labels, mixture_weights)
        """
        assert len(pc.shape) == 3, f"Expected (B, 3, N), got {pc.shape}"
        n_sampled_points = pc.shape[2]

        # For reconstruction, we want to use posterior mean (deterministic)
        output_encoder = self.encode(pc)

        g_sample, _, _ = self.quantizer(output_encoder)

        samples, labels = self.decoder.decode(g_sample, n_sampled_points)
        
        return samples, labels
    
    
    def sample(self, n_sampled_points, n_samples=1):
        """
        Generate new point clouds by sampling from the prior.
        
        Args:
            n_sampled_points: number of points per generated cloud
            n_samples: number of point clouds to generate
            device: device to generate samples on (defaults to model device)
            s
        Returns:
            Tuple of (prior_output, samples, labels, mixture_weights)
        """            
        # Set deterministic mode for consistent evaluation
        was_training = self.training
        if was_training:
            self.eval()
        try:
            g_sample = self.quantizer.sample(n_samples, device=self.device)

            samples, labels = self.decoder.decode(g_sample, n_sampled_points)

            return samples, labels
            
        finally:
            # Restore original training state
            if was_training:
                self.train()


    @torch.no_grad()
    def interpolate(self, pc1, pc2, n_steps=10, n_sampled_points=None):
        """
        Interpolate between two point clouds in latent space.
        
        Args:
            pc1: first point cloud (1, 3, N) or (3, N)
            pc2: second point cloud (1, 3, N) or (3, N)
            n_steps: number of interpolation steps
            n_sampled_points: points per generated cloud (defaults to input size)
            
        Returns:
            List of interpolated point clouds
        """
        # Ensure batch dimension
        if len(pc1.shape) == 2:
            pc1 = pc1.unsqueeze(0)
        if len(pc2.shape) == 2:
            pc2 = pc2.unsqueeze(0)
            
        if n_sampled_points is None:
            n_sampled_points = pc1.shape[2]

        # Encode both point clouds
        latent1 = self.encode(pc1)['g_posterior_samples']
        latent2 = self.encode(pc2)['g_posterior_samples']
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, n_steps, device=latent1.device)
        
        interpolated_clouds = []
        for alpha in alphas:
            # Linear interpolation in latent space
            latent_interp = (1 - alpha) * latent1 + alpha * latent2
            
            # Decode interpolated latent
            samples, _, _ = self.decoder.decode(latent_interp, n_sampled_points)
            interpolated_clouds.append(samples)
            
        return interpolated_clouds
    
    def get_model_info(self):
        """Get comprehensive model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters()) 
        prior_params = sum(p.numel() for p in self.latent_prior.parameters())
        posterior_params = sum(p.numel() for p in self.quantizer.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'prior_flow_parameters': prior_params,
            'posterior_parameters': posterior_params,
            'latent_dim': self.latent_dim,
            'model_device': str(next(self.parameters()).device)
        }
    

    def forward(self, p, g, n_sampled_points=None, warmup=False):
        p = p.transpose(1, 2)
        sampled_cloud_size = p.shape[2] if n_sampled_points is None else n_sampled_points
        print(f"Forward pass with sampled cloud size: {sampled_cloud_size}")

        latent, kl_loss = self.encode(g)
        
        recont_loss = self.decoder(p, latent, sampled_cloud_size, warmup)

        output = {
            "loss": recont_loss + kl_loss,
            "recont_loss": recont_loss,
            "kl_loss": kl_loss
        }

        return output