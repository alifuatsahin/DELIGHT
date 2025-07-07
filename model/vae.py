from .encoder import Encoder
from .decoder import Decoder
from modules.flows import RealNVPFlowCouple
from modules.layers import MLP

import torch
import torch.nn as nn

class LatentPriorFlow(nn.Module):
    def __init__(
        self,
        depth,
        local_feature_dim,
        global_feature_dim,
        weight_std=0.01,
    ):
        super().__init__()
        self.depth = depth
        self.local_feature_dim = local_feature_dim
        self.global_feature_dim = global_feature_dim
        self.weight_std = weight_std

        self.flows = nn.ModuleList([
            RealNVPFlowCouple(local_feature_dim, global_feature_dim, weight_std=self.weight_std, pattern=(i % 2))
            for i in range(depth)
        ])

    def forward(self, g, mode='direct'):
        """
        Apply the flow to the global features.
        Args:
            g: global features
            mode: 'direct' or 'inverse'
        Returns:
            gs: transformed global features (list)
            mus: means of the flows (list)
            logvars: log variances of the flows (list)
        """
        gs, mus, logvars = [], [], []
        
        if mode == 'direct':
            cur_g = g
            for flow in self.flows:
                buf = flow(cur_g, mode=mode)
                gs.extend(buf[0])
                mus.extend(buf[1])
                logvars.extend(buf[2])
                cur_g = gs[-1]  # Use last output as next input
                
        elif mode == 'inverse':
            cur_g = g
            for flow in reversed(self.flows):
                buf = flow(cur_g, mode=mode)
                gs = buf[0] + gs  # Prepend for inverse
                mus = buf[1] + mus
                logvars = buf[2] + logvars
                cur_g = gs[0] if gs else g  # Use first output as next input
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return gs, mus, logvars


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.latent_dim = cfg.model.latent_dim
        self.cfg = cfg  # Store config for debugging

        self.encoder = Encoder(cfg.model.input_dim)
        self.decoder = Decoder(cfg.model)

        self.latent_prior = LatentPriorFlow(
            depth=cfg.model.prior_flow_depth,
            local_feature_dim=cfg.model.prior_feat_dim,
            global_feature_dim=cfg.model.latent_dim,
            weight_std=0.01
        )

        self.latent_posterior = MLP(
            cfg.model.posterior_n_layers, self.encoder.out_features,
            cfg.model.latent_dim, deterministic=False,
            mu_weight_std=0.0033, mu_bias=0.0,
            logvar_weight_std=0.033, logvar_bias=0.0
        )

        self.latent_prior_mus = nn.Parameter(torch.Tensor(1, self.latent_dim))
        self.latent_prior_logvars = nn.Parameter(torch.Tensor(1, self.latent_dim))
        
        # Initialize parameters properly
        nn.init.zeros_(self.latent_prior_mus)
        nn.init.zeros_(self.latent_prior_logvars)
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick.
        
        Args:
            mean: mean values
            logvar: log variance values
            
        Returns:
            Sampled values (deterministic during eval mode)
        """
            
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, x):
        output = {}
        batch_size = x.shape[0]
        
        # Pre-expand parameters once for efficiency
        output['g_prior_mus'] = [self.latent_prior_mus.expand(batch_size, -1)]
        output['g_prior_logvars'] = [self.latent_prior_logvars.expand(batch_size, -1)]

        latent = self.encoder(x)

        # get posterior distribution from point cloud features
        output['g_posterior_mus'], output['g_posterior_logvars'] = self.latent_posterior(latent)
        output['g_posterior_samples'] = self.reparameterize(
            output['g_posterior_mus'], 
            output['g_posterior_logvars']
        )

        # train prior flow / auto-encoding task get prior distribution
        # g_prior_samples represents list of output transformations after couping layers
        buf_g = self.latent_prior(output['g_posterior_samples'], mode='inverse')
        # inverse training, the last layer is the g_posterior_samples computed from the input
        # point cloud, used for loss computation.
        output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]

        # g_prior_logvars returns the list of prior logvars generated after coupling layers
        # g_prior_mus returns the list of prior mus generated after coupling layers
        output['g_prior_mus'] += buf_g[1]
        output['g_prior_logvars'] += buf_g[2]

        return output
    
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

        if deterministic:
            g_sample = output_encoder['g_posterior_mus']
        else:
            g_sample = output_encoder['g_posterior_samples']

        samples, labels, mixture_weights_logits = self.decoder.decode(g_sample, n_sampled_points)
        
        return samples, labels, mixture_weights_logits
    
    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1, deterministic=False):
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
        device = next(self.parameters()).device
            
        # Set deterministic mode for consistent evaluation
        was_training = self.training
        if was_training:
            self.eval()

        try:
            # Pre-expand parameters once for efficiency
            prior_mus = self.latent_prior_mus.expand(n_samples, -1).to(device)
            prior_logvars = self.latent_prior_logvars.expand(n_samples, -1).to(device)
            
            output = {
                'g_prior_mus': [prior_mus],
                'g_prior_logvars': [prior_logvars]
            }

            # Generate initial sample (deterministic if requested)
            if deterministic:
                initial_sample = prior_mus  # Use mean without noise
            else:
                initial_sample = self.reparameterize(prior_mus, prior_logvars)
                
            output['g_prior_samples'] = [initial_sample]
            
            # Apply normalizing flows
            buf_g = self.latent_prior(output['g_prior_samples'][0], mode='direct')
            output['g_prior_samples'] += buf_g[0]
            output['g_prior_mus'] += buf_g[1]
            output['g_prior_logvars'] += buf_g[2]

            g_sample = output['g_prior_samples'][-1]
            samples, labels, mixture_weights_logits = self.decoder.decode(g_sample, n_sampled_points)

            return output, samples, labels, mixture_weights_logits
            
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
        posterior_params = sum(p.numel() for p in self.latent_posterior.parameters())
        
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
        sampled_cloud_size = p.shape[2] if n_sampled_points is None else n_sampled_points

        output_encoder = self.encode(g)
        
        g_sample = output_encoder['g_posterior_samples']
        output_decoder, mixture_weights_logits = self.decoder(p, g_sample, sampled_cloud_size, warmup)

        return output_encoder, output_decoder, mixture_weights_logits