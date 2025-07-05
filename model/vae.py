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
            g: transformed global features
            mus: means of the flows
            logvars: log variances of the flows
        """
        gs = []
        mus = []
        logvars = []
        for i in range(self.depth):
            if mode == 'direct':
                cur_g = g if i == 0 else gs[-1]
                buf = self.flows[i](cur_g, mode=mode)
                gs = gs + buf[0]
                mus = mus + buf[1]
                logvars = logvars + buf[2]
            elif mode == 'inverse':
                cur_g = g if i == 0 else gs[0]
                buf = self.flows[-(i + 1)](cur_g, mode=mode)
                gs = buf[0] + gs
                mus = buf[1] + mus
                logvars = buf[2] + logvars

        return gs, mus, logvars


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.latent_dim = cfg.model.latent_dim

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
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mean + eps * std

    def encode(self, x):
        output = {}
        
        output['g_prior_mus'] = [self.latent_prior_mus.expand(x.shape[0], self.latent_dim)]
        output['g_prior_logvars'] = [self.latent_prior_logvars.expand(x.shape[0], self.latent_dim)]

        latent = self.encoder(x)

        # get posterior distribution from point cloud features
        output['g_posterior_mus'], output['g_posterior_logvars'] = self.latent_posterior(latent)
        output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'],
                output['g_posterior_logvars'])

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
    
    def sample(self, n_sampled_points, n_samples=1):
        output = {}
        
        output['g_prior_mus'] = [self.latent_prior_mus.expand(n_samples, self.latent_dim)]
        output['g_prior_logvars'] = [self.latent_prior_logvars.expand(n_samples, self.latent_dim)]

        # generation task, get prior distribution
        output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]
        buf_g = self.latent_prior(output['g_prior_samples'][0], mode='direct')
        # direct transformation, the last layer is the predicted sample distribution
        output['g_prior_samples'] += buf_g[0]
        output['g_prior_mus'] += buf_g[1]
        output['g_prior_logvars'] += buf_g[2]

        g_sample = output['g_prior_samples'][-1]

        samples, labels, mixture_weights_logits = self.decoder.decode(g_sample, n_sampled_points)

        return output, samples, labels, mixture_weights_logits


    def forward(self, p, g, n_sampled_points=None, warmup=False):
        sampled_cloud_size = p.shape[2] if n_sampled_points is None else n_sampled_points

        output_encoder = self.encode(g)
        
        g_sample = output_encoder['g_posterior_samples']
        output_decoder, mixture_weights_logits = self.decoder(p, g_sample, sampled_cloud_size, warmup)

        return output_encoder, output_decoder, mixture_weights_logits