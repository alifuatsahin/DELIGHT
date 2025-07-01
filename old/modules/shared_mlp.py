import functools
import torch
import torch.nn as nn
from modules.swish import Swish

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1, cfg={}):
        assert(len(cfg) > 0), "Configuration dictionary must not be empty."
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
        elif dim == 2:
            conv = nn.Conv2d
        else:
            raise ValueError("Unsupported dimension: only 1D and 2D convolutions are supported.")
        self.bn_type = cfg.get('bn_type', 'adagn')
        bn = nn.GroupNorm if self.bn_type == 'groupnorm' else functools.partial(AdaGN, dim, cfg)

        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for out_ch in out_channels:
            layers.append(conv(in_channels, out_ch, 1))
            layers.append(bn(8, out_ch))
            layers.append(Swish())
            in_channels = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs):
        if self.bn_type == 'adagn':
            if len(inputs) == 1 and len(inputs[0]) == 4:
                inputs = inputs[0]
            if len(inputs) == 4:
                x, _, _, style = inputs
                for l in self.layers:
                    if isinstance(l, AdaGN):
                        x = l(x, style)
                    else:
                        x = l(x)
                return (x, *inputs[1:])
            elif len(inputs) == 2:
                x, style = inputs
                for l in self.layers:
                    if isinstance(l, AdaGN):
                        x = l(x, style)
                    else:
                        x = l(x)
                return x
            else:
                raise NotImplementedError("Unsupported dimension of the input: {}".format(len(inputs)))
        else:
            if isinstance(inputs, (list, tuple)):
                return (self.layers(inputs[0]), *inputs[1:])
            else:
                return self.layers(inputs)

