import torch.nn as nn
import torch
from collections import OrderedDict

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedDot(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features,
        n_channels,
        bias=False,
        init_weight=None, 
        init_bias=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_channels = n_channels
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.weight = nn.Parameter(torch.Tensor(n_channels, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_channels, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weight:
            nn.init.uniform_(self.weight.data, a=-self.init_weight, b=self.init_weight)
        else:
            nn.init.kaiming_uniform_(self.weight.data, a=0.)
        if self.bias is not None:
            if self.init_bias:
                nn.init.constant_(self.bias.data, self.init_bias)
            else:
                nn.init.constant_(self.bias.data, 0.)

    def forward(self, input):
        output = torch.matmul(self.weight, input.unsqueeze(1))
        if self.bias is not None:
            output.add_(self.bias.unsqueeze(0).unsqueeze(3))
        output.squeeze_(1)
        return output
    

class MLP(nn.Module):
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