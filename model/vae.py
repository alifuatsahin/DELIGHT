from .encoder import Encoder
from .decoder import Decoder
from modules.flows import RealNVPFlowCouple
from modules.layers import MLP

import torch
import torch.nn as nn

class LatentPriorFlow(nn.Module):
    def __init__(
        self,
        n_flows,
        local_feature_dim,
        global_feature_dim,
        weight_std=0.01,
    ):
        super().__init__()
        self.n_flows = n_flows
        self.local_feature_dim = local_feature_dim
        self.global_feature_dim = global_feature_dim
        self.weight_std = weight_std

        self.flows = nn.ModuleList([
            RealNVPFlowCouple(local_feature_dim, global_feature_dim, weight_std=self.weight_std, pattern=(i % 2))
            for i in range(n_flows)
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
        for i in range(self.n_flows):
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
    def __init__(self, cfg, mode='training'):
        super().__init__()
        decoder_cfg = cfg.decoder

        self.mode = mode
        self.latent_dim = cfg.latent_dim

        self.encoder = Encoder(cfg.latent_dim, cfg.input_dim)
        self.decoder = Decoder(decoder_cfg, mode=mode)

        self.latent_prior = LatentPriorFlow(
            n_flows=decoder_cfg.n_flows,
            local_feature_dim=decoder_cfg.feat_dim,
            global_feature_dim=decoder_cfg.latent_dim,
            weight_std=0.01
        )

        self.latent_posterior = MLP(
            cfg.posterior_n_layers, self.encoder.out_features,
            cfg.latent_dim, deterministic=False,
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

        if self.mode == 'training':
            posterior_feats = self.encoder(x)
            latent = torch.max(posterior_feats, dim=2)[0]

            # get posterior distribution from point cloud features
            output['g_posterior_mus'], output['g_posterior_logvars'] = self.latent_posterior(latent)
            output['g_posterior_samples'] = self.reparameterize(output['g_posterior_mus'],
                    output['g_posterior_logvars']) if self.mode == 'training' else output['g_posterior_mus']

            # train prior flow / auto-encoding task get prior distribution
            # g_prior_samples represents list of output transformations after couping layers
            buf_g = self.g_prior(output['g_posterior_samples'], mode='inverse')
            # inverse training, the last layer is the g_posterior_samples computed from the input
            # point cloud, used for loss computation.
            output['g_prior_samples'] = buf_g[0] + [output['g_posterior_samples']]

        elif self.mode == 'generating':
            # generation task, get prior distribution
            output['g_prior_samples'] = [self.reparameterize(output['g_prior_mus'][0], output['g_prior_logvars'][0])]
            buf_g = self.g_prior(output['g_prior_samples'][0], mode='direct')
            # direct transformation, the last layer is the predicted sample distribution
            output['g_prior_samples'] += buf_g[0]

        # g_prior_logvars returns the list of prior logvars generated after coupling layers
        # g_prior_mus returns the list of prior mus generated after coupling layers
        output['g_prior_mus'] += buf_g[1]
        output['g_prior_logvars'] += buf_g[2]

        return output
    
    def forward(self, p, g, n_sampled_points=None, labeled_samples=False, warmup=False):

        sampled_cloud_size = p.shape[2] if n_sampled_points is None else n_sampled_points

        output_encoder = self.encode(g)
        
        g_sample = output_encoder['g_posterior_samples'] if self.mode == 'training' else output_encoder['g_prior_samples'][-1]
        
        if labeled_samples:
            samples, labels, mixture_weights_logits = self.decode(p, g_sample, sampled_cloud_size, labeled_samples, warmup)
            return output_encoder, samples, labels, mixture_weights_logits
        else:
            output_decoder, mixture_weights_logits = self.decode(p, g_sample, sampled_cloud_size, labeled_samples, warmup)
            return output_encoder, output_decoder, mixture_weights_logits