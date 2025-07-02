import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from model.layers import SharedDot, Swish

class CondRealNVPFlow3D(nn.Module):
    def __init__(self, flow_feat_dim, latent_dim,
                 weight_std=0.01, warp_inds=[0],
                 centered_translation=False, eps=1e-6):
        super().__init__()
        self.flow_feat_dim = flow_feat_dim
        self.latent_dim = latent_dim
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = [0, 1, 2]
        self.centered_translation = centered_translation
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_sd0', SharedDot(len(self.keep_inds), self.flow_feat_dim, 1)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', SharedDot(self.flow_feat_dim, self.flow_feat_dim, 1)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.flow_feat_dim, affine=False))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd1_film_w0_swish', Swish()),
            ('mu_sd1_film_w1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd1_film_b0_swish', Swish()),
            ('mu_sd1_film_b1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', SharedDot(self.flow_feat_dim, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', SharedDot(len(self.keep_inds), self.flow_feat_dim, 1)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', SharedDot(self.flow_feat_dim, self.flow_feat_dim, 1)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.flow_feat_dim, affine=False))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd1_film_w0_swish', Swish()),
            ('logvar_sd1_film_w1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd1_film_b0_swish', Swish()),
            ('logvar_sd1_film_b1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', SharedDot(self.flow_feat_dim, len(self.warp_inds), 1, bias=True))
        ]))

        with torch.no_grad():
            self.T_logvar_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_w[-1].bias.data, 0.0)
            self.T_logvar_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0_cond_b[-1].bias.data, 0.0)
            self.T_logvar_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_1[-1].bias.data, 0.0)

    def forward(self, p, g, mode='direct'):
        logvar = torch.zeros_like(p)
        mu = torch.zeros_like(p)

        logvar[:, self.warp_inds, :] = nn.functional.softsign(self.T_logvar_1(
            torch.add(self.eps, torch.exp(self.T_logvar_0_cond_w(g).unsqueeze(2))) *
            self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        ))

        mu[:, self.warp_inds, :] = self.T_mu_1(
            torch.add(self.eps, torch.exp(self.T_mu_0_cond_w(g).unsqueeze(2))) *
            self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        )

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            p_out = torch.sqrt(torch.add(self.eps, torch.exp(logvar))) * p + mu
        elif mode == 'inverse':
            p_out = (p - mu) / torch.sqrt(torch.add(self.eps, torch.exp(logvar)))

        return p_out, mu, logvar


class CondRealNVPFlow3DTriple(nn.Module):
    def __init__(self, flow_feat_dim, latent_dim, weight_std=0.02, pattern=0, centered_translation=False):
        super().__init__()

        if pattern == 0:
            warp_inds_list = [[0], [1], [2]]
        elif pattern == 1:
            warp_inds_list = [[0, 1], [0, 2], [1, 2]]
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        self.nvps = nn.ModuleList([
            CondRealNVPFlow3D(
                flow_feat_dim, latent_dim,
                weight_std=weight_std,
                warp_inds=warp_inds,
                centered_translation=centered_translation
            )
            for warp_inds in warp_inds_list
        ])

    def forward(self, p, g, mode='direct'):
        if mode == 'direct':
            p1, mu1, logvar1 = self.nvps[0](p, g, mode=mode)
            p2, mu2, logvar2 = self.nvps[1](p1, g, mode=mode)
            p3, mu3, logvar3 = self.nvps[2](p2, g, mode=mode)
        elif mode == 'inverse':
            p3, mu3, logvar3 = self.nvps[2](p, g, mode=mode)
            p2, mu2, logvar2 = self.nvps[1](p3, g, mode=mode)
            p1, mu1, logvar1 = self.nvps[0](p2, g, mode=mode)

        return [p1, p2, p3], [mu1, mu2, mu3], [logvar1, logvar2, logvar3]
    
    
class RealNVPFlow(nn.Module):
    def __init__(self, n_features, g_n_features, weight_std=0.01, warp_inds=[0], eps=1e-6):
        super(RealNVPFlow, self).__init__()
        self.n_features = n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.warp_inds = warp_inds
        self.keep_inds = list(np.arange(g_n_features))
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        for ind in self.warp_inds:
            self.keep_inds.remove(ind)

        self.T_mu_0 = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(len(self.keep_inds), self.n_features, bias=False)),
            ('mu_mlp0_bn', nn.BatchNorm1d(self.n_features)),
            ('mu_mlp0_swish', Swish()),
            ('mu_mlp1', nn.Linear(self.n_features, len(self.warp_inds), bias=True))
        ]))
        with torch.no_grad():
            self.T_mu_0[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_mlp0', nn.Linear(len(self.keep_inds), self.n_features, bias=False)),
            ('logvar_mlp0_bn', nn.BatchNorm1d(self.n_features)),
            ('logvar_mlp0_swish', Swish()),
            ('logvar_mlp1', nn.Linear(self.n_features, len(self.warp_inds), bias=True))
        ]))
        with torch.no_grad():
            self.T_logvar_0[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_logvar_0[-1].bias.data, 0.0)

    def forward(self, g, mode='direct'):
        logvar = torch.zeros_like(g)
        mu = torch.zeros_like(g)

        logvar[:, self.warp_inds] = torch.log(torch.add(
            self.eps,
            torch.exp(self.T_logvar_0(g[:, self.keep_inds].contiguous()))
        ))
        mu[:, self.warp_inds] = self.T_mu_0(g[:, self.keep_inds].contiguous())

        logvar = logvar.contiguous()
        mu = mu.contiguous()

        if mode == 'direct':
            g_out = torch.exp(0.5 * logvar) * g + mu
        elif mode == 'inverse':
            g_out = torch.exp(-0.5 * logvar) * (g - mu)

        return g_out, mu, logvar


class RealNVPFlowCouple(nn.Module):
    def __init__(self, n_features, g_n_features, weight_std=0.01, pattern=0):
        super(RealNVPFlowCouple, self).__init__()
        self.n_features = n_features
        self.g_n_features = g_n_features
        self.weight_std = weight_std
        self.pattern = pattern

        if pattern == 0:
            self.nvp1 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[::2]))
            self.nvp2 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[1::2]))
        elif pattern == 1:
            self.nvp1 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[:g_n_features // 2]))
            self.nvp2 = RealNVPFlow(n_features, g_n_features,
                                    weight_std=weight_std, warp_inds=list(np.arange(g_n_features)[g_n_features // 2:]))

    def forward(self, g, mode='direct'):
        if mode == 'direct':
            g1, mu1, logvar1 = self.nvp1(g, mode=mode)
            g2, mu2, logvar2 = self.nvp2(g1, mode=mode)
        elif mode == 'inverse':
            g2, mu2, logvar2 = self.nvp2(g, mode=mode)
            g1, mu1, logvar1 = self.nvp1(g2, mode=mode)

        return [g1, g2], [mu1, mu2], [logvar1, logvar2]