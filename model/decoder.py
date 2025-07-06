import torch
import torch.nn as nn
import math
import numpy as np

from modules.flows import CondRealNVPFlow3DTriple
from modules.layers import MLP


class WeightsNetwork(MLP):
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
    def __init__(self, cfg):
        super().__init__()
        self.n_flows = cfg.n_flows
        self.depth = cfg.depth
        self.feat_dim = cfg.feat_dim
        self.latent_dim = cfg.latent_dim
        self.input_dim = cfg.input_dim

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
            in_features=self.latent_dim,
            out_features=self.n_flows,
            mu_weight_std=0.001,
            mu_bias=0.0
        )

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

    def get_weights(self, latent_vector, warmup=False):
        """
        Get the mixture weights for the decoder flows.
        Args:
            latent_feats: features to compute weights from
            warmup: whether to use warmup mode
        Returns:
            mixture_weights: computed weights
        """
        if warmup:
            return self.mixture_weights_logits.unsqueeze(0).expand(latent_vector.shape[0], self.n_flows)

        return self.mixture_weights_enc(latent_vector)
    
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
    
    def decode(self, latents, n_sampled_points):
        samples = []
        labels = []
        mixture_weights_logits_list = []

        for g in latents:
            g = g.unsqueeze(0)
            mixture_weights_logits = self.get_weights(g, warmup=False)
            # #when evaluation, each time, only one shape is inputed
            # assert g.shape[0] == 1

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
                flow_out = {}
                # for training/generation task
                flow_out['p_prior_mus'], flow_out['p_prior_logvars'] = self.point_prior(g)
                flow_out['p_prior_mus'] = [flow_out['p_prior_mus'].unsqueeze(2).expand(
                    g.shape[0], self.input_dim, n_sample_flow[i]
                )]
                flow_out['p_prior_logvars'] = [flow_out['p_prior_logvars'].unsqueeze(2).expand(
                    g.shape[0], self.input_dim, n_sample_flow[i]
                )]
                flow_out['p_prior_samples'] = [self.reparametrize(flow_out['p_prior_mus'][0], flow_out['p_prior_logvars'][0])]
                buf = decoder(flow_out['p_prior_samples'][0], g, mode='direct')
                flow_out['p_prior_samples'] += buf[0]
                flow_out['p_prior_mus'] += buf[1]
                flow_out['p_prior_logvars'] += buf[2]

                output.append(flow_out)

            sample = torch.zeros(g.shape[0], self.input_dim, n_sampled_points).to(g.device)
            label = torch.zeros(g.shape[0], n_sampled_points)
            for t in range(self.n_flows):
                #for each point, find its labels (generated by which flow)
                s = output[t]
                mask = flows_idx == t
                sample[:, :, mask] = s['p_prior_samples'][-1]
                label[:, mask] = t + 1
        
            samples.append(sample)
            labels.append(label)
            mixture_weights_logits_list.append(mixture_weights_logits)

        samples = torch.cat(samples, dim=0)
        labels = torch.cat(labels, dim=0)
        mixture_weights_logits = torch.cat(mixture_weights_logits_list, dim=0)

        return samples, labels, mixture_weights_logits

    def forward(self, p, g, n_sampled_points, warmup=False):
        """
        Decode the input features using the decoder flows. (During training)
        Args:
            p: input features
            g: additional conditioning features
            n_sampled_points: number of points to sample
            mode: 'training' or 'generating'
            warmup: whether to use warmup mode
        Returns:
            ps: decoded features
            mus: means of the flows
            logvars: log variances of the flows
        """
        mixture_weights_logits = self.get_weights(g, warmup=warmup)

        n_sample_flow = [n_sampled_points for _ in range(self.n_flows)]
            
        output = []

        for i, decoder in enumerate(self.decoder):
            #generate output parts for each flow decoder
            flow_out = {}
            # for training/generation task
            flow_out['p_prior_mus'], flow_out['p_prior_logvars'] = self.point_prior(g)
            flow_out['p_prior_mus'] = [flow_out['p_prior_mus'].unsqueeze(2).expand(
                g.shape[0], self.input_dim, n_sample_flow[i]
            )]
            flow_out['p_prior_logvars'] = [flow_out['p_prior_logvars'].unsqueeze(2).expand(
                g.shape[0], self.input_dim, n_sample_flow[i]
            )]
            buf = decoder(p, g, mode='inverse')
            flow_out['p_prior_samples'] = buf[0] + [p]
            flow_out['p_prior_mus'] += buf[1]
            flow_out['p_prior_logvars'] += buf[2]
                
            output.append(flow_out)

        return output, mixture_weights_logits