import math
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = {'fan_in', 'fan_out', 'fan_avg'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {valid_modes}.")
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out if mode == 'fan_out' else (fan_in + fan_out) / 2

def kaiming_uniform_(tensor, gain=1., mode='fan_in'):
    fan = _calculate_correct_fan(tensor, mode)

    var = gain / max(1., fan)
    bound = math.sqrt(3.0 * var)  # Calculate uniform bounds from variance
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
def variance_scaling_init(tensor, scale):
    return kaiming_uniform_(tensor, gain=1e-10 if scale == 0 else scale, mode='fan_avg')

def dense(in_channels, out_channels, init_scale=1.):
    """
    Creates a dense (fully connected) layer with custom initialization.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        init_scale (float): Scale for the initialization. Default is 1.0.

    Returns:
        nn.Linear: A fully connected layer with custom initialization.
    """
    layer = nn.Linear(in_channels, out_channels)
    variance_scaling_init(layer.weight, scale=init_scale)
    nn.init.zeros_(layer.bias)
    return layer