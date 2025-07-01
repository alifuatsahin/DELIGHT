import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import custom_bwd, custom_fwd

class BallQuery(nn.Module):
    def __init__(self, radius, nsample, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.include_coordinates = include_coordinates

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, point_xyz, center_xyz, features=None):
        """
        Args:
            point_xyz (torch.Tensor): Point cloud coordinates of shape (B, N, 3).
            center_xyz (torch.Tensor): Center coordinates of shape (B, M, 3).
            features (torch.Tensor, optional): Features associated with points of shape (B, N, C).
        Returns:
            torch.Tensor: Indices of points within the radius for each center, shape (B, M, nsample).
            torch.Tensor: Coordinates of the points within the radius, shape (B, M, nsample, 3).
            torch.Tensor: Center coordinates, shape (B, M, 3).
        """
        point_xyz = point_xyz.contiguous()
        center_xyz = center_xyz.contiguous()
        neighbor_indices = F.ball_query(
            point_xyz, center_xyz, self.radius, self.nsample
        )
        neighbor_xyz = F.grouping(
            point_xyz, neighbor_indices
        )
        neighbor_xyz = neighbor_xyz - center_xyz.unsqueeze(-1)

        if features is not None:
            assert self.include_coordinates, "No features provided for grouping."
            neighbor_features = neighbor_xyz

        else:
            neighbor_features = F.grouping(
                features, neighbor_indices
            )
            if self.include_coordinates:
                neighbor_features = torch.cat(
                    [neighbor_features, neighbor_xyz], dim=1
                )

        return neighbor_features

    @custom_bwd(device_type='cuda')
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)
    
    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')