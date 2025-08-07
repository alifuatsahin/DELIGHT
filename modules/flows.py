import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from .layers import SharedDot, Swish
from .RQS import unconstrained_RQS

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
            ('mu_sd0', nn.Conv1d(len(self.keep_inds), self.flow_feat_dim, kernel_size=1)),
            ('mu_sd0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd0_relu', nn.ReLU(inplace=True)),
            ('mu_sd1', nn.Conv1d(self.flow_feat_dim, self.flow_feat_dim, kernel_size=1)),
            ('mu_sd1_bn', nn.BatchNorm1d(self.flow_feat_dim))
        ]))

        self.T_mu_0_cond_w = nn.Sequential(OrderedDict([
            ('mu_sd1_film_w0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('mu_sd1_film_w0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd1_film_w0_swish', nn.SiLU()),
            ('mu_sd1_film_w1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_mu_0_cond_b = nn.Sequential(OrderedDict([
            ('mu_sd1_film_b0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('mu_sd1_film_b0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('mu_sd1_film_b0_swish', nn.SiLU()),
            ('mu_sd1_film_b1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_mu_1 = nn.Sequential(OrderedDict([
            ('mu_sd1_relu', nn.ReLU(inplace=True)),
            ('mu_sd2', nn.Conv1d(self.flow_feat_dim, len(self.warp_inds), kernel_size=1, bias=True))
        ]))

        with torch.no_grad():
            self.T_mu_0_cond_w[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_w[-1].bias.data, 0.0)
            self.T_mu_0_cond_b[-1].weight.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_0_cond_b[-1].bias.data, 0.0)
            self.T_mu_1[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.T_mu_1[-1].bias.data, 0.0)

        self.T_logvar_0 = nn.Sequential(OrderedDict([
            ('logvar_sd0', nn.Conv1d(len(self.keep_inds), self.flow_feat_dim, kernel_size=1)),
            ('logvar_sd0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd0_relu', nn.ReLU(inplace=True)),
            ('logvar_sd1', nn.Conv1d(self.flow_feat_dim, self.flow_feat_dim, kernel_size=1)),
            ('logvar_sd1_bn', nn.BatchNorm1d(self.flow_feat_dim))
        ]))

        self.T_logvar_0_cond_w = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_w0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('logvar_sd1_film_w0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd1_film_w0_swish', nn.SiLU()),
            ('logvar_sd1_film_w1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_logvar_0_cond_b = nn.Sequential(OrderedDict([
            ('logvar_sd1_film_b0', nn.Linear(self.latent_dim, self.flow_feat_dim, bias=False)),
            ('logvar_sd1_film_b0_bn', nn.BatchNorm1d(self.flow_feat_dim)),
            ('logvar_sd1_film_b0_swish', nn.SiLU()),
            ('logvar_sd1_film_b1', nn.Linear(self.flow_feat_dim, self.flow_feat_dim, bias=True))
        ]))

        self.T_logvar_1 = nn.Sequential(OrderedDict([
            ('logvar_sd1_relu', nn.ReLU(inplace=True)),
            ('logvar_sd2', nn.Conv1d(self.flow_feat_dim, len(self.warp_inds), kernel_size=1, bias=True))
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

        logvar_warp = nn.functional.softsign(self.T_logvar_1(
            torch.add(self.eps, F.softplus(self.T_logvar_0_cond_w(g).unsqueeze(2))) *
            self.T_logvar_0(p[:, self.keep_inds, :].contiguous()) + self.T_logvar_0_cond_b(g).unsqueeze(2)
        ))

        mu_warp = self.T_mu_1(
            torch.add(self.eps, F.softplus(self.T_mu_0_cond_w(g).unsqueeze(2))) *
            self.T_mu_0(p[:, self.keep_inds, :].contiguous()) + self.T_mu_0_cond_b(g).unsqueeze(2)
        )

        # Safe assignment with dtype matching
        logvar[:, self.warp_inds, :] = logvar_warp.to(logvar.dtype)
        mu[:, self.warp_inds, :] = mu_warp.to(mu.dtype)

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
    
class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, K = 5, B = 3, hidden_dim = 8, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0])
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0])
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

class CondLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )
    
    def forward(self, x):
        return self.network(x)

class CondNSFTriple3D(nn.Module):
    """
    Neural spline flow for 3D data with triple coupling layers.
    """
    def __init__(self, flow_feat_dim, latent_dim, K=5, B=3, n_features=3, eps=1e-6):
        super().__init__()
        self.K = K
        self.B = B
        self.indices = np.arange(n_features)
        self.register_buffer('eps', torch.from_numpy(np.array([eps], dtype=np.float32)))
        self.init_layers(flow_feat_dim, latent_dim)

    def init_layers(self, flow_feat_dim, latent_dim):
        """
        Initialize the layers for the flow.
        """
        self.W0 = nn.Sequential(
            SharedDot(len(self.indices)-1, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim),
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim, affine=False)
        )
        self.W1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, 1, 1, bias=True)
        )
        self.H0 = nn.Sequential(
            SharedDot(len(self.indices)-1, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim),
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim, affine=False)
        )
        self.H1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, 1, 1, bias=True)
        )
        self.D0 = nn.Sequential(
            SharedDot(len(self.indices)-1, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim),
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, flow_feat_dim, 1),
            nn.BatchNorm1d(flow_feat_dim, affine=False)
        )
        self.D1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SharedDot(flow_feat_dim, 1, 1, bias=True)
        )

        # Film conditioning layers
        self.gammaW = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)
        self.betaW = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)
        self.gammaH = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)
        self.betaH = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)
        self.gammaD = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)
        self.betaD = CondLayer(latent_dim, flow_feat_dim, hidden_dim=flow_feat_dim)

    def W_cond(self, x, context, keep_inds):
        """
        Film conditioning applied to the specified input with context.
        """
        gamma = self.gammaW(torch.add(self.eps, torch.exp(context)).unsqueeze(-1))
        beta = self.betaW(context).unsqueeze(-1)
        W_warp = self.W1(gamma * self.W0(x[:, keep_inds, :].contiguous()) + beta)
        W_warp = torch.softmax(W_warp, dim=2)
        return 2 * self.B * W_warp

    def H_cond(self, x, context, keep_inds):
        """
        Film conditioning applied to the specified input with context.
        """
        gamma = self.gammaH(torch.add(self.eps, torch.exp(context)).unsqueeze(-1))
        beta = self.betaH(context).unsqueeze(-1)
        H_warp = self.H1(gamma * self.H0(x[:, keep_inds, :].contiguous()) + beta)
        H_warp = torch.softmax(H_warp, dim=2)
        return 2 * self.B * H_warp
    
    def D_cond(self, x, context, keep_inds):
        """
        Film conditioning applied to the specified input with context.
        """
        gamma = self.gammaD(torch.add(self.eps, torch.exp(context)).unsqueeze(-1))
        beta = self.betaD(context).unsqueeze(-1)
        D_warp = self.D1(gamma * self.D0(x[:, keep_inds, :].contiguous()) + beta)
        D_warp = F.softplus(D_warp)
        return D_warp

    def flow_pass(self, x, context, warp_inds, keep_inds, inverse=False):
        """
        Perform a single flow pass through the network.
        """
        W = self.W_cond(x, context, keep_inds=keep_inds)
        H = self.H_cond(x, context, keep_inds=keep_inds)
        D = self.D_cond(x, context, keep_inds=keep_inds)

        warp_out, ld = unconstrained_RQS(
            x[:, warp_inds, :], W, H, D, inverse=inverse, tail_bound=self.B)

        return warp_out, ld

    def forward(self, x, context):
        log_det = torch.zeros(x.shape[0])

        for ind in self.indices:
            keep_inds = [i for i in self.indices if i != ind]
            out, ld = self.flow_pass(x, context, warp_inds=[ind], keep_inds=keep_inds)
            log_det += torch.sum(ld, dim=1)
            x = torch.cat([x[:, keep_inds, :], out], dim=1)

        return x, log_det

    def inverse(self, x, context):
        log_det = torch.zeros(x.shape[0])

        for ind in reversed(self.indices):
            keep_inds = np.delete(self.indices, np.where(self.indices == ind))
            out, ld = self.flow_pass(x, context, inverse=True, keep_inds=keep_inds)
            log_det += torch.sum(ld, dim=1)
            x = torch.cat([x[:, keep_inds, :], out], dim=1)

        return x, log_det
