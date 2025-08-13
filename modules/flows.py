import torch
import torch.nn as nn
from .layers import FiLMCond
from .attention import CrossAttention

class FlowBase(nn.Module):
    def __init__(self, n_layers, latent_dim, flow_dim=256, input_dim=3, cond_dim=64, cond='attn'):
        super().__init__()
        assert n_layers > 0, "Number of layers must be greater than 0"

        if cond == 'attn':
            self.cond = CrossAttention(input_dim, dim_head=cond_dim)
        elif cond == 'film':
            self.cond = FiLMCond(input_dim, latent_dim, cond_dim)

        self.features = nn.Sequential()
        self.features.add_module('input_proj', nn.Linear(input_dim, flow_dim, bias=False))
        self.features.add_module('input_bn', nn.BatchNorm1d(flow_dim))
        self.features.add_module('input_swish', nn.SiLU())

        for i in range(n_layers-1):
            self.features.add_module('mlp{}'.format(i), nn.Linear(flow_dim, flow_dim, bias=False))
            self.features.add_module('mlp{}_bn'.format(i), nn.BatchNorm1d(flow_dim))
            self.features.add_module('mlp{}_swish'.format(i), nn.SiLU())

    def forward(self, x, context):
        # Apply the conditional layer
        x_cond = self.cond(x, context)

        # Pass through the feature layers
        x = self.features(x_cond)
        return x