from .kl import Quantizer as KL
from .softvq import Quantizer as SoftVQ

def get_quantizer(cfg, input_dim):
    """
    Factory function to get the appropriate quantizer based on the configuration.
    
    Args:
        cfg: Configuration object containing quantizer settings.
        input_dim: Dimension of the input features.
        
    Returns:
        An instance of the specified quantizer class.
    """
    quantizer_type = cfg.quantizer
    if quantizer_type == 'kl':
        return KL(cfg, input_dim)
    elif quantizer_type == 'softvq':
        return SoftVQ(cfg, input_dim)
    else:
        available_types = ['kl', 'softvq']
        raise ValueError(
            f"Unknown quantizer type: '{quantizer_type}'. "
            f"Available types: {available_types}"
        )
    
__all__ = ['KL', 'SoftVQ', 'get_quantizer']