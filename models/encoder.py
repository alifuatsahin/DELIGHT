import torch
import torch.nn as nn 
from typing import Tuple, List, Optional
from modules.pvcnn2 import create_pointnet2_sa_components 

class Encoder(nn.Module):
    """
    Point cloud encoder using PointNet++ architecture with set abstraction layers.
    
    Processes point clouds through hierarchical feature learning with attention
    and squeeze-excitation mechanisms for robust global feature extraction.
    """
    
    # Default architecture configuration
    DEFAULT_SA_BLOCKS = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 64]]],
        [[64, 1, 16], [256, 0.2, 32, [64, 128]]],
        [[128, 1, 16], [128, 0.4, 64, [128, 512, 1024]]],
    ]
    
    def __init__(
        self, 
        input_dim: int = 3, 
        extra_feature_channels: int = 0,
        sa_blocks: Optional[List] = None,
        use_attention: bool = True,
        use_se: bool = True,
        force_attention: int = 1
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
        self.input_dim = input_dim
        self.extra_feature_channels = extra_feature_channels
        self.use_attention = use_attention
        self.use_se = use_se
        self.force_attention = force_attention
        
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
                extra_feature_channels, 
                input_dim=input_dim, 
                embed_dim=0, 
                force_att=self.force_attention,
                use_att=self.use_attention, 
                with_se=self.use_se
            )
        
        self.out_features = channels_sa_features
        self.layers = nn.ModuleList(layers) 
        self.voxel_dimensions = [block[1][-1][-1] for block in self.sa_blocks]

        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

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
            features, xyz, _ = layer((features, xyz, None))

        # features: (B, D, N) -> (B, D)
        return features, xyz
    
    def get_architecture_info(self) -> dict:
        """
        Get detailed architecture information.
        
        Returns:
            Dictionary with architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_dim': self.input_dim,
            'output_features': self.out_features,
            'num_sa_blocks': len(self.sa_blocks),
            'voxel_dimensions': self.voxel_dimensions,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_attention': self.use_attention,
            'use_se': self.use_se,
            'extra_feature_channels': self.extra_feature_channels
        }
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            List of feature maps from each SA layer
        """
        x = x.transpose(1, 2)
        xyz = x[:, :3, :]
        features = x
        
        feature_maps = [features]
        
        for layer in self.layers:
            features, xyz, _ = layer((features, xyz, None))
            feature_maps.append(features)
        
        return feature_maps
    
    @torch.no_grad()
    def analyze_receptive_field(self, x: torch.Tensor) -> dict:
        """
        Analyze the receptive field of the encoder.
        
        Args:
            x: Input point cloud (B, N, 3)
            
        Returns:
            Receptive field statistics
        """
        feature_maps = self.get_feature_maps(x)
        
        stats = {
            'input_points': x.shape[1],
            'layer_point_counts': [fm.shape[-1] for fm in feature_maps],
            'layer_feature_dims': [fm.shape[1] for fm in feature_maps],
            'compression_ratios': []
        }
        
        for i in range(1, len(feature_maps)):
            ratio = feature_maps[i-1].shape[-1] / feature_maps[i].shape[-1]
            stats['compression_ratios'].append(ratio)
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of the encoder."""
        info = self.get_architecture_info()
        return (f"Encoder(input_dim={info['input_dim']}, "
                f"output_features={info['output_features']}, "
                f"sa_blocks={info['num_sa_blocks']}, "
                f"params={info['total_parameters']:,})")


