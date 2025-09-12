import torch
import torch.nn as nn 
from typing import Tuple, List, Optional
from modules import create_pointnet2_sa_components, create_mlp_components

class Surrogate(nn.Module):
    """    
    Processes point clouds through hierarchical feature learning with attention
    and squeeze-excitation mechanisms for robust global feature extraction.
    """
    
    # Default architecture configuration
    DEFAULT_SA_BLOCKS = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 64]]],
        [[64, 1, 16], [256, 0.2, 32, [64, 128]]],
        [[128, 1, 16], [128, 0.4, 64, [128, 256]]],
        (None, (16, 0.8, 32, (256, 256, 256))),
    ]
    
    def __init__(
        self, 
        cfg,
    ):
        """
        Initialize the encoder.
        
        Args:
            input_dim: Input point dimension (default: 3 for xyz)
            extra_feature_channels: Additional feature channels beyond xyz
            sa_blocks: Custom set abstraction blocks configuration
            use_attention: Whether to use attention mechanisms
            use_se: Whether to use squeeze-excitation
            force_attention: Force attention on all layers (0=auto, 1=force)
        """
        super().__init__()
        
        # Store configuration
        self.input_dim = cfg.surrogate.input_dim
        self.extra_feature_channels = cfg.surrogate.extra_feature_channels
        self.use_attention = cfg.surrogate.use_attention
        self.use_se = cfg.surrogate.use_se
        self.force_attention = cfg.surrogate.force_attention
        self.dropout = cfg.surrogate.dropout
        self.width_multiplier = cfg.surrogate.width_multiplier
        self.sa_blocks = cfg.getattr('surrogate', 'sa_blocks', None)

        # Use provided or default architecture
        if sa_blocks is None:
            sa_blocks = self.DEFAULT_SA_BLOCKS
        self.sa_blocks = sa_blocks
        
        # Validate configuration
        self._validate_config()
        
        # Create PointNet++ components
        layers, sa_in_channels, channels_sa_features, _ = \
            create_pointnet2_sa_components(
                sa_blocks, 
                self.extra_feature_channels, 
                input_dim=self.input_dim, 
                emb_dim=None, 
                force_att=self.force_attention,
                use_att=self.use_attention, 
                with_se=self.use_se
            )
        
        self.out_features = channels_sa_features
        self.layers = nn.ModuleList(layers) 
        self.voxel_dimensions = [block[1][-1][-1] for block in self.sa_blocks]

        layers, _ = create_mlp_components(
                in_channels=channels_sa_features, 
                out_channels=[128, self.dropout, 1], # was 0.5
                classifier=True, dim=2, width_multiplier=self.width_multiplier,
                emb_dim=None)
        self.classifier = nn.ModuleList(layers)

        # Initialize parameters
        self._initialize_parameters()
    
    def _validate_config(self):
        """Validate encoder configuration."""
        assert self.input_dim > 0, "Input dimension must be positive"
        assert self.extra_feature_channels >= 0, "Extra feature channels must be non-negative"
        assert len(self.sa_blocks) > 0, "Must have at least one set abstraction block"
    
    def _initialize_parameters(self):
        """Initialize encoder parameters with proper schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args: 
            x: Input point cloud (B, N, 3) or (B, N, input_dim)
            
        Returns: 
            Global features (B, out_features)
        """
        # Validate input shape
        assert len(x.shape) == 3, f"Expected (B, N, D) input, got {x.shape}"
        assert x.shape[2] >= self.input_dim, f"Expected at least {self.input_dim} dimensions, got {x.shape[2]}"
        
        # Transpose to (B, D, N) format expected by PointNet++
        x = x.transpose(1, 2)  # (B, N, D) -> (B, D, N)
        
        # Extract coordinates and features
        xyz = x[:, :3, :]  # Always use first 3 dims as coordinates
        features = x  # Use all dimensions as features
        
        # Pass through set abstraction layers
        for layer in self.layers:
            features, xyz, _, _ = layer(features, xyz, None)

        for l in self.classifier:
            features = l(features)

        return features

