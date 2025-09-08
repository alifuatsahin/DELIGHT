import functools
import torch.nn as nn
import torch
import third_party.pvcnn.functional as F
from torch.amp import custom_fwd, custom_bwd

from .attention import TransformerBlock, CondTransformerBlock

class PointNetSequential(nn.Sequential):
    def forward(self, x, coords, emb=None, context=None, point_coords=None, point_feats=None):
        for layer in self:
            if isinstance(layer, SharedMLP):
                x = layer(x, emb)
            elif isinstance(layer, (PointNetAModule, PointNetSAModule)):
                x, coords, emb = layer(x, coords, emb)
            elif isinstance(layer, PointNetFPModule):
                x, coords, emb = layer(x, coords, point_feats, point_coords, emb=emb)
            else:
                x, coords, emb, context = layer(x, coords, emb, context)
        return x, coords, emb, context

def dense(in_channels, out_channels):
    lin = nn.Linear(in_channels, out_channels)
    nn.init.kaiming_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin

class AdaGN(nn.Module):
    '''
    adaptive group normalization
    '''
    def __init__(self, ndim, n_channel, emb_dim=None):
        """
        ndim: dim of the input features 
        n_channel: number of channels of the inputs 
        ndim_style: channel of the style features 
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.ndim = ndim
        self.n_channel = n_channel
        self.out_dim = n_channel * 2
        self.norm = nn.GroupNorm(8, n_channel)
        in_channel = n_channel 
        if emb_dim is not None:
            self.emd = dense(emb_dim, n_channel*2)
            self.emd.bias.data[:in_channel] = 1
            self.emd.bias.data[in_channel:] = 0
        
    def forward(self, input, emb=None):
        result = self.norm(input)
        if emb is None:
            assert self.emb_dim is None, "expecting style embedding"
            return result
        emb = self.emd(emb)
        if self.ndim == 3: #B,D,V,V,V
            emb = emb.view(emb.shape[0], -1, 1, 1, 1) # 5D 
        elif self.ndim == 2: # B,D,N,1 
            emb = emb.view(emb.shape[0], -1, 1, 1) # 4D 
        elif self.ndim == 1: # B,D,N
            emb = emb.view(emb.shape[0], -1, 1) # 4D 
        else:
            raise NotImplementedError

        factor, bias = emb.chunk(2, 1)
        result = result * (1 + factor) + bias  
        return result 

class SE3d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return input * self.fc(input.mean(-1).mean(-1).mean(-1)).view(input.shape[0], input.shape[1], 1, 1, 1)
    
class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates

    @custom_bwd(device_type='cuda')
    def backward(self, *args, **kwargs):
        return super().backward(*args, **kwargs)

    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, point_coords, center_coords, point_features=None):
        point_coords = point_coords.contiguous()
        center_coords = center_coords.contiguous()
        neighbor_indices = F.ball_query(center_coords, point_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(point_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - center_coords.unsqueeze(-1)

        if point_features is None:
            assert self.include_coordinates, "No features for grouping"
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(point_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_features, neighbor_coordinates], dim=1)
        return neighbor_features
    
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d
        
        bn = functools.partial(AdaGN, dim, emb_dim=emb_dim)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        
        for oc in out_channels:
            layers.append(conv(in_channels, oc, 1))
            layers.append(bn(oc))
            layers.append(nn.SiLU())
            in_channels = oc
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, emb=None):
        for l in self.layers:
            if isinstance(l, AdaGN):
                x = l(x, emb)
            else:
                x = l(x)
        return x

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(
                dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps
            ) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0

        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        if features is None:
            return features, norm_coords
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, resolution, emb_dim=None,
                 normalize=1, eps=0, with_se=False,
                 add_point_feat=True, attention=False,
                 context_dim=None, dropout=0.1):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(
            resolution,
            normalize=normalize,
            eps=eps
        )
        NormLayer = functools.partial(AdaGN, 3, emb_dim=emb_dim)
        voxel_layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size, stride=1,
                      padding=kernel_size // 2),
            NormLayer(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels,
                      kernel_size, stride=1,
                      padding=kernel_size // 2),
            NormLayer(out_channels)
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.ModuleList(voxel_layers)
        if attention:
            if context_dim is None:
                self.attn = TransformerBlock(out_channels, dim_head=64)
            else:
                self.attn = CondTransformerBlock(out_channels, dim_head=64, context_dim=context_dim)
        else:
            self.attn = None
        if add_point_feat:
            self.point_features = SharedMLP(in_channels, out_channels, emb_dim=emb_dim)
        self.add_point_feat = add_point_feat

    def forward(self, x, coords_input, emb=None, context=None):
        if coords_input.shape[1] > 3:
            coords_input = coords_input[:, :3]
        else:
            coords = coords_input

        assert (x.shape[0] == coords.shape[0]
                ), f'Get feat: {x.shape} and {coords.shape}'
        assert (x.shape[2] == coords.shape[2]
                ), f'Get feat: {x.shape} and {coords.shape}'
        assert (coords.shape[1] == 3
                ), f'Expect coords: (B,3,Npoint), get: {coords.shape}'

        voxel_features_4d, voxel_coords = self.voxelization(x, coords)
        r = self.resolution
        
        for voxel_layers in self.voxel_layers:
            if isinstance(voxel_layers, AdaGN):
                voxel_features_4d = voxel_layers(voxel_features_4d, emb)
            else:
                voxel_features_4d = voxel_layers(voxel_features_4d)

        voxel_features = F.trilinear_devoxelize(voxel_features_4d, voxel_coords, r, self.training)

        fused_features = voxel_features
        if self.add_point_feat:
            fused_features = fused_features + self.point_features(x, emb)

        if self.attn is not None:
            fused_features = self.attn(fused_features, context=context)
        
        return fused_features, coords_input, emb, context
    
class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None, include_coordinates=True):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]

        mlps = []
        total_out_channels = 0

        for _out_channels in out_channels:
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0),
                          out_channels=_out_channels, dim=1, emb_dim=emb_dim)
            )
            total_out_channels += _out_channels[-1]
        
        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x, coords, emb=None):
        if self.include_coordinates:
            x = torch.cat([x, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(mlp(x, emb).max(dim=-1, keepdim=True).values)
            return torch.cat(features_list, dim=1), coords, emb
        else:
            return self.mlps[0](x, emb).max(dim=-1, keepdim=True).values, coords, emb

class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, emb_dim=None, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors]
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers, mlps = [], []
        total_out_channels = 0

        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(
                    radius=_radius, 
                    num_neighbors=_num_neighbors, 
                    include_coordinates=include_coordinates
                    )
            )
            mlps.append(
                SharedMLP(
                    in_channels=in_channels + (3 if include_coordinates else 0),
                    out_channels=_out_channels,
                    dim=2,
                    emb_dim=emb_dim
                )
            )
            total_out_channels += _out_channels[-1]
        
        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, x, coords, emb=None):
        if coords.shape[1] > 3:
            coords = coords[:, :3]
        
        center_coords = F.furthest_point_sample(coords, self.num_centers)
        features_list = []
        c = 0
        for grouper, mlp in zip(self.groupers, self.mlps):
            c += 1
            grouper_output = grouper(coords, center_coords, x)
            features_list.append(
                mlp(grouper_output, emb).max(dim=-1, keepdim=False).values
            )

        if len(features_list) > 1:
            return torch.cat(features_list, dim=1), center_coords, emb
        else:
            return features_list[0], center_coords, emb

class PointNetFPModule(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=None):
        super().__init__()
        self.mlp = SharedMLP(in_channels, out_channels, dim=1, emb_dim=emb_dim)

    def forward(self, feats, coords, point_feats, point_coords, emb=None):
        interpolated_features = F.nearest_neighbor_interpolate(point_coords, coords, feats)
        interpolated_features = torch.cat([interpolated_features, point_feats], dim=1)
        return self.mlp(interpolated_features, emb), point_coords, emb