import torch
import torch.nn as nn
import math
from collections import OrderedDict

from .layers import Swish
from modules.flows import CondRealNVPFlow3DTriple

class PriorNetwork(nn.Module):
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


class WeightsNetwork(PriorNetwork):
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
    def __init__(
        self,
        n_flows,
        depth,
        feat_dim,
        latent_dim,
        weight_std=0.01
    ):
        super().__init__()
        self.n_flows = n_flows
        self.depth = depth
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.weight_std = weight_std

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
            n_layers=3,
            in_features=feat_dim,
            latent_space_size=latent_dim,
            mu_weight_std=0.001,
            mu_bias=0.0
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

    def decode(self, p, g, mode='direct', warmup=False):
        """
        Decode the input features using the decoder flows.
        Args:
            p: input features
            g: additional conditioning features
            mode: 'direct' or 'inverse'
            warmup: whether to use warmup mode
        Returns:
            ps: decoded features
            mus: means of the flows
            logvars: log variances of the flows
        """
        mixture_weights = self.get_weights(p, warmup=warmup)
        ps, mus, logvars = [], [], []
        
        for i, decoder in enumerate(self.decoder):
            cur_p, cur_mus, cur_logvars = decoder(p, g, mode=mode)
            ps.append(cur_p)
            mus.append(cur_mus)
            logvars.append(cur_logvars)

        return ps, mus, logvars, mixture_weights