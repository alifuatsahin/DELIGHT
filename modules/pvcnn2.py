"""
copied and modified from source: 
    https://github.com/alexzhou907/PVD/blob/9747265a5f141e5546fd4f862bfa66aa59f1bd33/model/pvcnn_generation.py 
    and functions under 
    https://github.com/alexzhou907/PVD/tree/9747265a5f141e5546fd4f862bfa66aa59f1bd33/modules 
"""
import copy
import functools
from loguru import logger
from einops import rearrange
import torch.nn as nn
import torch
import numpy as np
import third_party.pvcnn.functional as F
from torch.amp import custom_fwd, custom_bwd
from modules.layers import Swish

import xformers.ops as xops
import math


class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
    def __repr__(self):
        return f"SE({self.channel}, {self.channel})" 
    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(inputs.shape[0], inputs.shape[1], 1, 1, 1)

class LinearAttention(nn.Module): 
    """
    copied and modified from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L159 
    """
    def __init__(self, dim, heads = 4, hidden_dim=4*32, verbose=True): 
        super().__init__()
        self.heads = heads
        dim_head = hidden_dim // heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1) 

    def forward(self, x):
        '''
        Args:
            x: torch.tensor (B,C,N), C=num-channels, N=num-points 
        Returns:
            out: torch.tensor (B,C,N)
        '''
        x = x.unsqueeze(-1) # add w dimension
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        out = self.to_out(out)
        out = out.squeeze(-1) # B,C,N,1 -> B,C,N
        return out 

class GateLinearAttentionNoSilu(nn.Module):

    def __init__(self, dim, num_heads=4, hidden_dim=4*32):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkvo = nn.Conv2d(dim, hidden_dim * 4, 1)
        self.elu = nn.ELU()
        self.proj = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor):
        '''
            x: (B, C, N), C=num-channels, N=num-points
        Returns:
            out: torch.tensor (B, C, N)
        '''
        x = x.unsqueeze(-1)  # add w dimension
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.hidden_dim, :, :]
        o = qkvo[:, 3*self.hidden_dim:, :, :]

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0 # (b n l d)

        q_mean = q.mean(dim=-2, keepdim=True) # (b n 1 d)
        eff = self.scale * q_mean @ k.transpose(-1, -2) # (b n 1 l)
        eff = torch.softmax(eff, dim=-1).transpose(-1, -2) # (b n l 1)
        k = k * eff * (H*W)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # (b n l 1)
        kv = (k.transpose(-2, -1) * ((H*W) ** -0.5)) @ (v * ((H*W) ** -0.5)) # (b n d d)

        res = q @ kv * z # (b n l d)
        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        out = self.proj(res * o)
        out = out.squeeze(-1)  # B, C, N, 1 -> B, C, N
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4, hidden_dim=4*32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert dim % n_heads == 0, "in_channels must be divisible by n_heads"
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1)
        self.attn = EfficientQKVAttention(n_heads)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # x: (B, C, N)
        x = x.unsqueeze(-1) # add w dimensio
        qkv = self.to_qkv(x)  # (B, 3*C, N)
        out = self.attn(qkv)  # (B, C, N)
        out = self.to_out(out)
        out = out.squeeze(-1)  # B, C, N, 1 -> B, C, N
        return out
    
class EfficientQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        B, C, H, W = qkv.shape
        assert C % (3 * self.n_heads) == 0
        ch = C // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()  # [B, H*W, n_heads, ch]
        k = k.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()
        scale = 1 / math.sqrt(ch)
        y = xops.memory_efficient_attention(q, k, v, scale=scale)
        y = y.permute(0, 2, 3, 1).reshape(B, self.n_heads * ch, H * W).unsqueeze(-1)
        return y

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
    def forward(self, points_coords, centers_coords, points_features=None):
        # input: BCN, BCN 
        # returns: 
        # neighbor_features: B,D(+3),Ncenter 
        points_coords = points_coords.contiguous()
        centers_coords = centers_coords.contiguous()
        neighbor_indices = F.ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        neighbor_coordinates = F.grouping(points_coords, neighbor_indices)
        neighbor_coordinates = neighbor_coordinates - centers_coords.unsqueeze(-1)

        if points_features is None:
            assert self.include_coordinates, 'No Features For Grouping'
            neighbor_features = neighbor_coordinates
        else:
            neighbor_features = F.grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                neighbor_features = torch.cat([neighbor_coordinates, neighbor_features], dim=1)
        return neighbor_features

    def extra_repr(self):
        return 'radius={}, num_neighbors={}{}'.format(
            self.radius, self.num_neighbors, ', include coordinates' if self.include_coordinates else '')

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim==1:
            conv = nn.Conv1d
        else:
            conv = nn.Conv2d
        bn = nn.GroupNorm 
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.append( conv(in_channels, oc, 1)) 
            layers.append(bn(8, oc))
            layers.append(nn.SiLU()) 
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)

class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        # features: B,D,N
        # coords:   B,3,N 
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(
                dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 +
                                         self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        if features is None:
            return features, norm_coords
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(
            self.r,
            ', normalized eps = {}'.format(self.eps) if self.normalize else '')

class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
        kernel_size, resolution, 
        normalize=1, eps=0, with_se=False, 
        add_point_feat=True, attention=False, 
        dropout=0.1, verbose=True 
        ):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution,
                                         normalize=normalize,
                                         eps=eps)
        # For each PVConv we use (Conv3d, GroupNorm(8), Swish, dropout, Conv3d, GroupNorm(8), Attention) 
        voxel_layers = [
            nn.Conv3d(in_channels, 
                      out_channels,
                      kernel_size, stride=1,
                      padding=kernel_size // 2), 
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels,
                        kernel_size, stride=1,
                        padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels)
            ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        if attention:
            # self.attn = LinearAttention(out_channels, verbose=verbose)
            # self.attn = GateLinearAttentionNoSilu(out_channels)
            self.attn = SelfAttentionBlock(out_channels)
        else:
            self.attn = None
        if add_point_feat:
            self.point_features = SharedMLP(in_channels, out_channels) #, **mlp_kwargs)
        self.add_point_feat = add_point_feat

    def forward(self, inputs):  
        '''
        Args: 
            inputs: tuple of features and coords 
                features: B,feat-dim,num-points 
                coords:   B,3, num-points 
        Returns:
            fused_features: in (B,out-feat-dim,num-points)
            coords        : in (B, 3, num_points); same as the input coords
        '''
        features = inputs[0] 
        coords_input = inputs[1]
        time_emb = inputs[2]
        ## features, coords_input, time_emb = inputs
        if coords_input.shape[1] > 3:
            coords = coords_input[:,:3] # the last 3 dim are other point attributes if any  
        else:
            coords = coords_input
        assert (features.shape[0] == coords.shape[0]
                ), f'get feat: {features.shape} and {coords.shape}'
        assert (features.shape[2] == coords.shape[2]
                ), f'get feat: {features.shape} and {coords.shape}'
        assert (coords.shape[1] == 3
                ), f'expect coords: B,3,Npoint, get: {coords.shape}'
        # features: B,D,N; point_features  
        # coords:   B,3,N 
        voxel_features_4d, voxel_coords = self.voxelization(features, coords)
        r = self.resolution 
        B = coords.shape[0]
        voxel_features_4d = self.voxel_layers(voxel_features_4d) 
        voxel_features = F.trilinear_devoxelize(voxel_features_4d, voxel_coords,
                                                r, self.training)

        fused_features = voxel_features 
        if self.add_point_feat:
            fused_features = fused_features + self.point_features(features)
        if self.attn is not None:
            fused_features = self.attn(fused_features)
        if time_emb is None:
            time_emb = {'voxel_features_4d': voxel_features_4d, 'resolution': self.resolution, 'training': self.training}  
        return fused_features, coords_input, time_emb #inputs[2]


class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, include_coordinates=True):
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
                          out_channels=_out_channels, dim=1)
            )
            total_out_channels += _out_channels[-1]

        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        features, coords, time_emb = inputs
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(mlp(features).max(dim=-1, keepdim=True).values)
            return torch.cat(features_list, dim=1), coords, time_emb
        else:
            return self.mlps[0](features).max(dim=-1, keepdim=True).values, coords, time_emb

    def extra_repr(self):
        return f'out_channels={self.out_channels}, include_coordinates={self.include_coordinates}'


class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()

        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(radius=_radius, num_neighbors=_num_neighbors, 
                    include_coordinates=include_coordinates)
            )
            # logger.info('create MLP: in_channel={}, out_channels={}',
            #        in_channels + (3 if include_coordinates else 0),_out_channels)
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0) ,
                          out_channels=_out_channels, dim=2)
            )
            total_out_channels += _out_channels[-1]

        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def forward(self, inputs):
        # features, coords, _ = inputs
        features = inputs[0] 
        coords = inputs[1]  # B3N 
        if coords.shape[1] > 3:
            coords = coords[:,:3]

        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        # centers_coords: B,D,N
        S = centers_coords.shape[-1]
        time_emb = inputs[2] 
        time_emb = time_emb[:,:,:S] if \
            time_emb is not None and type(time_emb) is not dict \
            else time_emb  

        features_list = []
        c = 0
        for grouper, mlp in zip(self.groupers, self.mlps):
            c += 1
            grouper_output = grouper(coords, centers_coords, features)
            features_list.append(
                    mlp(grouper_output
                        ).max(dim=-1).values
                    )
        if len(features_list) > 1:
            return torch.cat(features_list, dim=1), centers_coords, time_emb
        else:
            return features_list[0], centers_coords, time_emb

    def extra_repr(self):
        return f'num_centers={self.num_centers}, out_channels={self.out_channels}'


# class PointNetFPModule(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

#     def forward(self, inputs):
#         if len(inputs) == 4:
#             points_coords, centers_coords, centers_features, time_emb = inputs
#             points_features = None
#         else:
#             points_coords, centers_coords, centers_features, points_features, time_emb = inputs
#         interpolated_features = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
#         if points_features is not None:
#             interpolated_features = torch.cat(
#                 [interpolated_features, points_features], dim=1
#             )
#         if time_emb is not None:
#             B,D,S = time_emb.shape 
#             N = points_coords.shape[-1]
#             time_emb = time_emb[:,:,0:1].expand(-1,-1,N) 
#         return self.mlp(interpolated_features), points_coords, time_emb

def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8,out_channels), nn.SiLU())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


# def create_pointnet_components(blocks, in_channels, embed_dim, with_se=False, normalize=True, eps=0,
#                                width_multiplier=1, voxel_resolution_multiplier=1, verbose=True):
#     r, vr = width_multiplier, voxel_resolution_multiplier

#     layers, concat_channels = [], 0
#     c = 0
#     for k, (out_channels, num_blocks, voxel_resolution) in enumerate(blocks):
#         out_channels = int(r * out_channels)
#         for p in range(num_blocks):
#             attention = k % 2 == 0 and k > 0 and p == 0
#             if voxel_resolution is None:
#                 block = SharedMLP
#             else:
#                 block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution), attention=attention,
#                                           with_se=with_se, normalize=normalize, eps=eps, verbose=verbose)

#             if c == 0:
#                 layers.append(block(in_channels, out_channels))
#             else:
#                 layers.append(block(in_channels+embed_dim, out_channels))
#             in_channels = out_channels
#             concat_channels += out_channels
#             c += 1
#     return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, 
        input_dim=3, 
        embed_dim=64, use_att=False, force_att=0,
        dropout=0.1, with_se=False, normalize=True, eps=0, has_temb=1,
        width_multiplier=1, voxel_resolution_multiplier=1, verbose=True):
    """
    Returns: 
        in_channels: the last output channels of the sa blocks 
    """
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + input_dim 

    sa_layers, sa_in_channels = [], []
    c = 0
    num_centers = None
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_blocks = []

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = ( (c) % 2 == 0 and use_att and p == 0 ) or (force_att and c > 0)
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(
                        PVConv, kernel_size=3, 
                        resolution=int(vr * voxel_resolution), attention=attention,
                        dropout=dropout,
                        with_se=with_se, 
                        normalize=normalize, eps=eps, verbose=verbose)

                if c == 0:
                    sa_blocks.append(block(in_channels, out_channels))
                elif k ==0:
                    sa_blocks.append(block(in_channels+embed_dim*has_temb, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels

        if sa_configs is not None:
            num_centers, radius, num_neighbors, out_channels = sa_configs
            _out_channels = []
            for oc in out_channels:
                if isinstance(oc, (list, tuple)):
                    _out_channels.append([int(r * _oc) for _oc in oc])
                else:
                    _out_channels.append(int(r * oc))
            out_channels = _out_channels
            if num_centers is None:
                block = PointNetAModule
            else:
                block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                          num_neighbors=num_neighbors) 
            sa_blocks.append(block(in_channels=extra_feature_channels+(embed_dim*has_temb if k==0 else 0 ), 
                out_channels=out_channels,
                include_coordinates=True))
            in_channels = extra_feature_channels = sa_blocks[-1].out_channels 
        c += 1

        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


# def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, embed_dim=64, use_att=False,
#                                 dropout=0.1, has_temb=1, 
#                                 with_se=False, normalize=True, eps=0,
#                                 width_multiplier=1, voxel_resolution_multiplier=1,
#                                 verbose=True):
#     r, vr = width_multiplier, voxel_resolution_multiplier

#     fp_layers = []
#     c = 0

#     for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
#         fp_blocks = []
#         out_channels = tuple(int(r * oc) for oc in fp_configs)
#         fp_blocks.append(
#             PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim*has_temb, 
#                 out_channels=out_channels)
#         )
#         in_channels = out_channels[-1]

#         if conv_configs is not None:
#             out_channels, num_blocks, voxel_resolution = conv_configs
#             out_channels = int(r * out_channels)
#             for p in range(num_blocks):
#                 attention = (c+1) % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
#                 if voxel_resolution is None:
#                     block = SharedMLP
#                 else:
#                     block = functools.partial(PVConv, kernel_size=3, 
#                             resolution=int(vr * voxel_resolution), attention=attention,
#                             dropout=dropout,
#                             with_se=with_se, # with_se_relu=True,
#                             normalize=normalize, eps=eps,
#                             verbose=verbose)

#                 fp_blocks.append(block(in_channels, out_channels))
#                 in_channels = out_channels
#         if len(fp_blocks) == 1:
#             fp_layers.append(fp_blocks[0])
#         else:
#             fp_layers.append(nn.Sequential(*fp_blocks))

#         c += 1

#     return fp_layers, in_channels


