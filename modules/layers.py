import torch
import torch.nn as nn
from collections import OrderedDict
from .attention import TransformerBlock

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

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
class AttnBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dim_head,
        n_heads,
        dropout=0.0,
        out_channels=None,
        use_xformers=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_xformers = use_xformers

        self.in_layers = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, 1),
        )

        self.cross_attn = TransformerBlock(
            dim=self.out_channels,
            context_dim=self.emb_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            use_xformers=self.use_xformers,
            use_pos_emb=True
        )

        self.out_norm = nn.BatchNorm1d(self.out_channels)

        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, context):
        h = self.in_layers(x)
        h = self.out_norm(h)
        h = self.cross_attn(h, context)
        h = self.out_layers(h)
        return self.skip_connection(x) + h