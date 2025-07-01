import torch
import torch.nn as nn
import third_party.pvcnn.functional as F

class Voxelization(nn.module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.resolution = resolution
        self.normalize = normalize
        self.eps = eps
    
    def forward(self, features, xyz):
        """
        Args:
            features (torch.Tensor): Features of shape (B, N, C).
            xyz (torch.Tensor): Coordinates of points of shape (B, N, 3).
        Returns:
            torch.Tensor: Voxelized features of shape (B, R, R, R, C) where R is the resolution.
        """
        xyz = xyz.detach()
        norm_xyz = xyz - xyz.mean(2, keepdim=True)
        if self.normalize:
            norm_xyz = norm_xyz / (norm_xyz.norm(
                dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_xyz = (norm_xyz + 1) / 2.0
        norm_xyz = torch.clamp(norm_xyz * self.resolution, 0, self.resolution - 1)
        vox_xyz = torch.round(norm_xyz).to(torch.int32)
        if features is not None:
            return features, norm_xyz
        return F.avg_voxelize(features, vox_xyz, self.resolution), norm_xyz
    
    def extra_repr(self):
        return 'resolution={}{}'.format(
            self.resolution,
            ', normalized eps = {}'.format(self.eps) if self.normalize else '')