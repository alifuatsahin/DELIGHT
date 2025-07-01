import torch.nn as nn
from modules.linear_attention import LinearAttention
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.swish import Swish
from modules.se import SE3d
import third_party.pvcnn.functional as F

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 resolution, normalize=1, eps=0, with_se=False,
                 add_point_feat=True, attention=False,
                 dropout=0.1, verbose=True
                    ):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize, eps)
        
        voxel_layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels,
                      out_channels, 
                      kernel_size=1, stride=1,
                      padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels)
            ]
        
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)

        if attention:
            self.attention = LinearAttention(out_channels, verbose=verbose)
        else:
            self.attention = None

        if add_point_feat:
            self.point_feat = SharedMLP(in_channels, out_channels)
        
        self.add_point_feat = add_point_feat

    def forward(self, inputs):
        """
        Args:
            inputs: tuple of (features, xyz) or just features.
        Returns:
            out: voxelized features after applying convolution and attention.
        """
        features = inputs[0]
        xyz_input = inputs[1]
        time_emb = inputs[2]

        if xyz_input.shape[1] >= 3:
            xyz = xyz_input[:, :3]
        else:
            xyz = xyz_input

        assert (features.shape[0] == xyz.shape[0]
                ), "Batch size of features and xyz must match."
        assert (features.shape[2] == xyz.shape[2]
                ), "Number of points in features and xyz must match."
        assert (xyz.shape[1] == 3
                ), "XYZ coordinates must have shape (B, N, 3)."
        
        voxel_features_4d, voxel_xyz = self.voxelization(features, xyz)
        r = self.resolution
        voxel_features_4d = self.voxel_layers(voxel_features_4d)
        voxel_features = F.trilinear_devoxelize(
            voxel_features_4d, voxel_xyz, r, self.training)
        
        fused_features = voxel_features
        if self.add_point_feat:
            fused_features = fused_features + self.point_features(features)
        if self.attention is not None:
            fused_features = self.attention(fused_features)
        if time_emb is not None:
            time_emb = {'voxel_features_4d': voxel_features_4d,
                        'training': self.training,
                        'resolution': self.resolution}