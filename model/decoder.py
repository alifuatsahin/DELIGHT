import torch
import torch.nn as nn
import math
import numpy as np
from collections import OrderedDict

from .layers import Swish
from modules.flows import CondRealNVPFlow3DTriple


class PointPriorNetwork(nn.Module):
    def __init__(
        self,
        n_layers,
        in_features,
        latent_dim,
        mu_weight_std=0.001,
        mu_bias=0.0,
        deterministic=False,
        logvar_weight_std=0.01, 
        logvar_bias=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.mu_weight_std = mu_weight_std
        self.mu_bias = mu_bias
        self.deterministic = deterministic

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_swish'.format(i), Swish())

        self.mus = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(in_features, latent_dim, bias=True))
        ]))

        with torch.no_grad():
            self.mus[-1].weight.data.normal_(std=mu_weight_std)
            nn.init.constant_(self.mus[-1].bias.data, mu_bias)

        if not self.deterministic:
            self.logvars = nn.Sequential(OrderedDict([
                ('logvar_mlp0', nn.Linear(in_features, latent_dim, bias=True))
            ]))
            with torch.no_grad():
                self.logvars[-1].weight.data.fill_(logvar_weight_std)
                nn.init.constant_(self.logvars[-1].bias.data, logvar_bias)

    def forward(self, input):
        if self.n_layers > 0:
            features = self.features(input)
        else:
            features = input

        if self.deterministic:
            mus = self.mus(features)

            return mus, None
        else:
            mus = self.mus(features)
            logvars = self.logvars(features)

            return mus, logvars


class WeightsNetwork(PointPriorNetwork):
    def forward(self, input):
        """
        Compute the weights for the decoder flows.
        Args:
            input: features to compute weights from
        Returns:
            weights: computed weights
        """
        mus, _ = super().forward(input)
        weights = nn.functional.log_softmax(mus, dim=1)

        return weights

class DecBlock(nn.Module):
    def __init__(
        self,
        depth,
        feat_dim,
        latent_dim,
        weight_std=0.01
    ):
        super().__init__()
        self.depth = depth
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.weight_std = weight_std

        self.layers = nn.ModuleList(
            [CondRealNVPFlow3DTriple(feat_dim, latent_dim,
                                    weight_std=self.weight_std, pattern=(i % 2)) for i in range(self.depth)]
        )
        
    def forward(self, p, g, mode='direct'):
        ps = []
        mus = []
        logvars = []
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
    def __init__(self, cfg, mode='training'):
        super().__init__()
        self.n_flows = cfg.n_flows
        self.depth = cfg.depth
        self.feat_dim = cfg.feat_dim
        self.latent_dim = cfg.latent_dim

        self.mode = mode # default mode

        self.flow_depth, self.feat_dim = self._get_decoder_params(min_feat_dim=4)

        self.mixture_weights_logits = torch.nn.Parameter(torch.zeros(self.n_flows), requires_grad=True)

        self.decoder = nn.ModuleList([
            DecBlock(self.flow_depth,
                    self.feat_dim,
                    self.latent_dim,
                    weight_std=0.01) 
                    for _ in range(self.n_flows)
                    ])
        
        self.mixture_weights_enc = WeightsNetwork(
            n_layers=cfg.weight_n_layers,
            in_features=self.feat_dim,
            latent_space_size=self.latent_dim,
            mu_weight_std=0.001,
            mu_bias=0.0
        )

        self.point_prior = PointPriorNetwork(
            n_layers=cfg.point_prior_n_layers,
            in_features=self.latent_dim,
            latent_dim=cfg.point_dim,
            mu_weight_std=0.001,
            mu_bias=0.0,
            deterministic=False,
            logvar_weight_std=0.01,
            logvar_bias=0.0
        )

    def _get_decoder_params(self, min_feat_dim=4):
        """
        Decide feature size and number of coupling layers under a parameter budget.
        Returns:
            flow_depth: coupling layers in each decoder flow
            feat_dim: feature size in each decoder flow
        """
        n = self.n_flows
        if n == 1:
            return self.depth, self.feat_dim

        # Compute flow_depth without modifying self.depth
        flow_depth = math.ceil(self.depth / math.sqrt(n))
        feat_dim = self.feat_dim
        max_param_count = self.get_param_count_for(flow_depth, feat_dim)

        # Try reducing feat_dim until under budget
        current_count = max_param_count
        while current_count > max_param_count and feat_dim > min_feat_dim:
            feat_dim -= 1
            current_count = self.get_param_count_for(flow_depth, feat_dim)

        return flow_depth, feat_dim

    def get_param_count_for(self, flow_depth, feat_dim):
        count_CondRealNVPFlow3D = 18 * feat_dim + 4 * feat_dim * self.latent_dim + 6 * feat_dim**2
        count_CondRealNVPFlow3DTriple = 3 * count_CondRealNVPFlow3D
        total_count = flow_depth * count_CondRealNVPFlow3DTriple * self.n_flows
        return total_count

    def get_weights(self, latent_feats, warmup=False):
        """
        Get the mixture weights for the decoder flows.
        Args:
            latent_feats: features to compute weights from
            warmup: whether to use warmup mode
        Returns:
            mixture_weights: computed weights
        """
        if warmup:
            return self.mixture_weights_logits.unsqueeze(0).expand(latent_feats.shape[0], self.n_flows)

        return self.mixture_weights_enc(latent_feats)
    
    def reparametrize(self, mus, logvars):
        """
        Reparameterization trick to sample from the latent space.
        Args:
            mus: means of the flows
            logvars: log variances of the flows
        Returns:
            samples: sampled features
        """
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        
        return eps.mul(std).add_(mus)
    
    def one_flow_decode(self, p, g, decoder, n_sampled_points):
        """
        Decode the input features using one flow decoder.
        Args:
            p: input features
            g: additional conditioning features
            decoder: one flow decoder
            n_sampled_points: number of points to sample
        Returns:
            output: decoded features
        """
        output = {}
        # for training/generation task
        output['p_prior_mus'], output['p_prior_logvars'] = self.point_prior(g)
        output['p_prior_mus'] = [output['p_prior_mus'].unsqueeze(2).expand(
            g.shape[0], self.p_latent_space_size, n_sampled_points
        )]
        output['p_prior_logvars'] = [output['p_prior_logvars'].unsqueeze(2).expand(
            g.shape[0], self.p_latent_space_size, n_sampled_points
        )]

        if self.mode == 'training':
            #train decoder flow
            buf = decoder(p, g, mode='inverse')
            output['p_prior_samples'] = buf[0] + [p]
        else:
            # for evaluation
            output['p_prior_samples'] = [self.reparameterize(output['p_prior_mus'][0], output['p_prior_logvars'][0])]
            buf = decoder(output['p_prior_samples'][0], g, mode='direct')
            output['p_prior_samples'] += buf[0]
        output['p_prior_mus'] += buf[1]
        output['p_prior_logvars'] += buf[2]

        return output

    def decode(self, p, g, n_sampled_points, warmup=False, labeled_samples=False):
        """
        Decode the input features using the decoder flows.
        Args:
            p: input features
            g: additional conditioning features
            n_sampled_points: number of points to sample
            mode: 'direct' or 'inverse'
            warmup: whether to use warmup mode
        Returns:
            ps: decoded features
            mus: means of the flows
            logvars: log variances of the flows
        """
        mixture_weights_logits = self.get_weights(p, warmup=warmup)

        if self.mode == 'training':
            n_sample_flow = [n_sampled_points for _ in range(self.n_flows)]

        else:
            #when evaluation, each time, only one shape is inputed
            assert p.shape[0] == 1

            #computes the probabilities of all flows
            logits_exp = np.exp(mixture_weights_logits[0].detach().cpu().numpy())
            probs = logits_exp / logits_exp.sum()

            #for each flow, randomly choose certain number of points based on its probability
            flows_idx = np.random.choice(range(self.n_flows), size=n_sampled_points, p=probs)

            #masks designs the labels
            masks = []
            for t in range(self.n_flows):
                mask = flows_idx == t
                masks.append(mask.sum())

            n_sample_flow = masks
            
        output = []
        for i, decoder in enumerate(self.decoder):
            #generate output parts for each flow decoder
            one_decoder = self.one_flow_decode(p, g, decoder, n_sample_flow[i])
            output.append(one_decoder)

        if labeled_samples:     #when for evaluation
            samples = torch.zeros_like((p))
            labels = torch.zeros(p.size(0), p.size(2))
            for t in range(self.n_components):
                #for each point, find its labels (generated by which flow)
                s = output[t]
                mask = flows_idx == t
                samples[:, :, mask] = s['p_prior_samples'][-1]
                labels[:, mask] = t + 1
            return samples, labels, mixture_weights_logits
        else:
            return output, mixture_weights_logits