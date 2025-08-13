import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MLP(nn.Module):
    def __init__(
        self,
        n_layers,
        in_features,
        out_features,
        mu_weight_std=0.001,
        mu_bias=0.0,
        deterministic=False,
        logvar_weight_std=0.001, 
        logvar_bias=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight_std = mu_weight_std
        self.mu_bias = mu_bias
        self.deterministic = deterministic

        if n_layers > 0:
            self.features = nn.Sequential()
            for i in range(n_layers):
                self.features.add_module('mlp{}'.format(i), nn.Linear(in_features, in_features, bias=False))
                self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(in_features))
                self.features.add_module('mlp{}_swish'.format(i), nn.SiLU())

        self.mus = nn.Sequential(OrderedDict([
            ('mu_mlp0', nn.Linear(in_features, out_features, bias=True))
        ]))

        with torch.no_grad():
            self.mus[-1].weight.data.normal_(std=mu_weight_std)
            nn.init.constant_(self.mus[-1].bias.data, mu_bias)

        if not self.deterministic:
            self.logvars = nn.Sequential(OrderedDict([
                ('logvar_mlp0', nn.Linear(in_features, out_features, bias=True))
            ]))
            with torch.no_grad():
                self.logvars[-1].weight.data.normal_(std=logvar_weight_std)
                nn.init.constant_(self.logvars[-1].bias.data, logvar_bias)

    def forward(self, input, warmup=False):
        if warmup:
            out = torch.zeros(input.shape[0], self.out_features, device=input.device)
            return out, out
        else:
            features = self.features(input)
            if self.deterministic:
                mus = self.mus(features)

                return mus
            else:
                mus = self.mus(features)
                logvars = self.logvars(features)

                return mus, logvars
        
class StandartGaussian(nn.Module):
    def __init__(self, out_features, mu=0.0, logvar=0.0):
        super().__init__()
        self.mu = mu
        self.logvar = logvar
        self.out_features = out_features

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mu + eps * std
    
    def forward(self, features, *args, **kwargs):
        """
        Forward pass for the StandartGaussian module.
        """
        out = torch.zeros(features.shape[0], self.out_features, device=features.device)
        return out, out
    

class FiLMCond(nn.Module):
    def __init__(self, input_dim, latent_dim, feat_dim, weight_std=0.001, bias=0.0):
        super().__init__()
        self.cond = nn.Sequential([
            nn.Conv1d(input_dim, feat_dim, kernel_size=1),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feat_dim, feat_dim, kernel_size=1),
        ])

        self.gamma = nn.Sequential(OrderedDict([
            nn.Linear(latent_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim, bias=True)
        ]))

        self.beta = nn.Sequential(OrderedDict([
            nn.Linear(latent_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, feat_dim, bias=True)
        ]))

        with torch.no_grad():
            self.cond[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.cond[-1].bias.data, bias)
            self.gamma[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.gamma[-1].bias.data, bias)
            self.beta[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.beta[-1].bias.data, bias)

    def forward(self, x, context):
        if context.dim() > 2:
            context = context.view(context.size(0), -1).contiguous()

        g = torch.add(F.softplus(self.gamma(context).unsqueeze(-1)), 1e-6)  # Ensure gamma is positive
        b = self.beta(context).unsqueeze(-1)
        out = g * self.cond(x) + b
        return out