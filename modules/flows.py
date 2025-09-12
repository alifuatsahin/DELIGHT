import torch
import torch.nn as nn
from collections import OrderedDict
import math
from einops import repeat

from .attention import SpatialTransformerBlock, CondTransformerBlock, AttentionBlock, LocalTransformerBlock, AttentionBlock2
from .layers import CondResBlock, ResBlock, ResBlock2
from .pvcnn import SharedMLP, PointNetSAModule, PointNetFPModule
from serialization import encode
from modules import create_pointnet2_fp_modules, create_pointnet2_sa_components, create_mlp_components

from loguru import logger

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: a (B, D, N) Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None, :, None]  # (B, half, N)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    else:
        embedding = repeat(timesteps, 'b n -> b d n', d=dim)
    return embedding

class PointSequential(nn.Sequential):
    """
    A sequential module that passes point cloud coordinates to the children that
    support it as an extra input.
    """

    def forward(self, x, coords=None, context=None):
        for layer in self:
            if isinstance(layer, (AttentionBlock, AttentionBlock2)):
                x = layer(x, coords, context)
            elif isinstance(layer, LocalTransformerBlock):
                x = layer(x, coords)
            else:
                x = layer(x)
        return x

class CondSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings or context to the children that
    support it as an extra input.
    """

    def forward(self, x, context, emb=None):
        for layer in self:
            if isinstance(layer, CondResBlock):
                x = layer(x, context)
            elif isinstance(layer, (CondTransformerBlock, SpatialTransformerBlock)):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class FlowBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.e_dim = cfg.softvq.e_dim
        self.input_dim = cfg.input_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Conv1d(self.t_emb_ch, t_emb_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(t_emb_dim, t_emb_dim, kernel_size=1),
        )
        self.to_in = nn.Conv1d(self.input_dim + t_emb_dim, self.width, kernel_size=1)

        layers = []

        for _ in range(self.depth):
            layers.append(CondResBlock(
                channels=self.width,
                emb_channels=self.latent_dim,
                dropout=0.0,
                out_channels=self.width,
                use_scale_shift_norm=True
            ))
        self.hidden_blocks = CondSequential(*layers)

        self.to_out = nn.Conv1d(self.width, self.input_dim, kernel_size=1)

        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

    def serialize(self, pc, grid_size=0.01, depth=16):
        idx = torch.randint(len(self.orders), (1,)).item()
        _, order, inverse = encode(pc, grid_size=grid_size, depth=depth, order=self.orders[idx])
        order = order.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        inverse = inverse.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        return order, inverse

    def forward(self, x, t, context):
        context = context.view(context.size(0), -1).contiguous() if context.dim() > 2 else context
        order, inverse = self.serialize(x)
        x = torch.gather(x, 1, order).transpose(2, 1).contiguous()
        t = torch.gather(t, 1, order[:, :, 0]).contiguous()
        t_emb = timestep_embedding(t, self.t_emb_ch)
        emb = self.time_embed(t_emb)
        h = torch.cat([x, emb], dim=1)  # Concatenate time embedding
        h = self.to_in(h)
        h = self.hidden_blocks(h, context)
        x = self.to_out(h)
        x = x.transpose(2, 1).contiguous()
        return torch.gather(x, 1, inverse).contiguous()
    
class PriorBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.prior.depth
        self.width = cfg.prior.width
        self.e_dim = cfg.vae.softvq.e_dim
        self.latent_dim = cfg.vae.latent_dim
        self.t_emb_ch = cfg.prior.t_emb_ch
        self.dim_head = cfg.prior.dim_head
        self.n_heads = cfg.prior.n_heads
        self.use_xformers = cfg.prior.use_xformers_attention

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Conv1d(self.t_emb_ch, t_emb_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(t_emb_dim, t_emb_dim, kernel_size=1),
        )
        # self.to_in = nn.Conv1d(self.e_dim + t_emb_dim, self.width, kernel_size=1)
        self.to_in = nn.Conv1d(self.e_dim, self.width, kernel_size=1)

        layers = []

        for _ in range(self.depth):
            layers.append(CondResBlock(
                channels=self.width,
                emb_channels=t_emb_dim,
                dropout=0.0,
                out_channels=self.width,
                use_scale_shift_norm=True
            ))
            layers.append(SpatialTransformerBlock(
                dim=self.width,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=True,
            ))
        self.hidden_blocks = CondSequential(*layers)

        self.to_out = nn.Conv1d(self.width, self.e_dim, kernel_size=1)

    def forward(self, x, t):
        x = x.transpose(1, 2).contiguous()
        t_emb = timestep_embedding(t, self.t_emb_ch)
        emb = self.time_embed(t_emb)
        emb = emb.view(emb.size(0), -1).contiguous()
        # h = torch.cat([x, emb], dim=1)  # Concatenate time embedding
        h = x
        h = self.to_in(h)
        h = self.hidden_blocks(h, emb)
        x = self.to_out(h)
        return x.transpose(1, 2).contiguous()

class ExpBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.input_dim = cfg.input_dim
        self.e_dim = cfg.softvq.e_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.dim_head = 64
        self.n_heads = 4
        self.use_xformers = True

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Conv1d(self.input_dim, self.width, kernel_size=1)

        layers = []

        for _ in range(self.depth):
            layers.append(CondResBlock(
                channels=self.width,
                emb_channels=t_emb_dim,
                dropout=0.0,
                out_channels=self.width,
                use_scale_shift_norm=True
            ))
            layers.append(CondTransformerBlock(
                dim=self.width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=False,
            ))
        self.hidden_blocks = CondSequential(*layers)

        self.to_out = nn.Conv1d(self.width, self.input_dim, kernel_size=1)
        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

    def serialize(self, pc, grid_size=0.01, depth=16):
        idx = torch.randint(len(self.orders), (1,)).item()
        _, order, inverse = encode(pc, grid_size=grid_size, depth=depth, order=self.orders[idx])
        order = order.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        inverse = inverse.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        return order, inverse

    def forward(self, x, t, context):
        x = x.transpose(1, 2).contiguous()
        t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
        emb = self.time_embed(t_emb)
        x = self.to_in(x)
        x = self.hidden_blocks(x, context=emb, emb=context)
        x = self.to_out(x)
        return x.transpose(1, 2).contiguous()

class Exp2Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.input_dim = cfg.input_dim
        self.e_dim = cfg.softvq.e_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.dim_head = 64
        self.n_heads = 8
        self.use_xformers = True

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Conv1d(self.input_dim + t_emb_dim, self.width, kernel_size=1)

        layers = []

        for _ in range(self.depth):
            layers.append(ResBlock(
                channels=self.width,
                dropout=0.0,
                out_channels=self.width,
            ))
            layers.append(CondTransformerBlock(
                dim=self.width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=False,
            ))
        self.hidden_blocks = CondSequential(*layers)

        self.to_out = nn.Conv1d(self.width, self.input_dim, kernel_size=1)
        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

    def serialize(self, pc, grid_size=0.01, depth=16):
        idx = torch.randint(len(self.orders), (1,)).item()
        _, order, inverse = encode(pc, grid_size=grid_size, depth=depth, order=self.orders[idx])
        order = order.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        inverse = inverse.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
        return order, inverse

    def forward(self, x, t, context):
        x = x.transpose(1, 2).contiguous()
        t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
        emb = self.time_embed(t_emb).unsqueeze(-1).expand(x.size(0), -1, x.size(2)).contiguous()
        x = torch.concat([x, emb], dim=1)  # Concatenate time embedding
        x = self.to_in(x)
        x = self.hidden_blocks(x, context=emb, emb=context)
        x = self.to_out(x)
        return x.transpose(1, 2).contiguous()
    
class Exp3Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.input_dim = cfg.input_dim
        self.e_dim = cfg.softvq.e_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.dim_head = 64
        self.n_heads = 4
        self.use_xformers = True

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Linear(self.input_dim + t_emb_dim, self.width)

        layers = []

        for _ in range(self.depth):
            layers.append(ResBlock2(
                channels=self.width,
                dropout=0.0,
                out_channels=self.width,
            ))
            layers.append(AttentionBlock2(
                dim=self.width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=True,
            ))
        self.hidden_blocks = PointSequential(*layers)

        self.to_out = nn.Linear(self.width, self.input_dim)

    def forward(self, x, t, context):
        context = context.transpose(1, 2).contiguous()
        coords = x[:, :, :3].contiguous()  # B N 3
        t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
        emb = self.time_embed(t_emb).unsqueeze(1).expand(x.size(0), x.size(1), -1).contiguous()
        x = torch.concat([x, emb], dim=-1)  # Concatenate time embedding
        x = self.to_in(x)
        x = self.hidden_blocks(x, coords=coords, context=context)
        return self.to_out(x)

class Exp4Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.input_dim = cfg.input_dim
        self.e_dim = cfg.softvq.e_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.dim_head = 64
        self.n_heads = 4
        self.use_xformers = True

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Linear(self.input_dim + t_emb_dim, self.width)

        layers = []

        for _ in range(self.depth):
            layers.append(LocalTransformerBlock(
                dim=self.width,
                n_heads=2,
                dim_head=64,
                use_pos_emb=True,
                patch_size=16,
            ))
            layers.append(AttentionBlock(
                dim=self.width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=False,
            ))
        self.hidden_blocks = PointSequential(*layers)

        self.to_out = nn.Linear(self.width, self.input_dim)

    def forward(self, x, t, context):
        context = context.transpose(1, 2).contiguous()
        coords = x[:, :, :3].contiguous()  # B N 3
        t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
        emb = self.time_embed(t_emb).unsqueeze(1).expand(x.size(0), x.size(1), -1).contiguous()
        x = torch.concat([x, emb], dim=-1)  # Concatenate time embedding
        x = self.to_in(x)
        x = self.hidden_blocks(x, coords=coords, context=context)
        return self.to_out(x)

class Exp5Base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.width = cfg.flow.width
        self.input_dim = cfg.input_dim
        self.e_dim = cfg.softvq.e_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.strides = (4, 2, 2)
        self.channel_mult = 2
        self.dim_head = 64
        self.n_heads = 4
        self.use_xformers = True
        radius = 0.1
        seq_len = 2048
        num_neighbors = (32, 32, 64)

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Linear(self.input_dim + t_emb_dim, self.width)

        layers = []

        widths = [self.width * (self.channel_mult ** i) for i in range(self.depth)]
        radiuses = [radius * (i + 1) for i in range(self.depth)]

        for width, radius, num_neighbor, stride in zip(widths, radiuses, num_neighbors, self.strides):
            layers.append(AttentionBlock2(
                dim=width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=16,
            ))
            layers.append(PointNetSAModule(
                num_centers=seq_len // stride,
                radius=radius,
                num_neighbors=num_neighbor,
                in_channels=width,
                out_channels=width * self.channel_mult,
            ))
        self.down_blocks = PointSequential(*layers)

        self.bottleneck = AttentionBlock2(
                dim=width * self.channel_mult,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=16,
            )

        layers = []

        for width in reversed(widths):
            layers.append(PointNetFPModule(
                in_channels=width * self.channel_mult + width,
                out_channels=width,
            ))
            layers.append(AttentionBlock2(
                dim=width,
                context_dim=self.e_dim,
                dim_head=self.dim_head,
                n_heads=self.n_heads,
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=16,
            ))
        self.up_blocks = PointSequential(*layers)

        self.to_out = nn.Linear(self.width, self.input_dim)

    def forward(self, x, t, context):
        context = context.transpose(1, 2).contiguous()
        coords = x[:, :, :3].contiguous()  # B N 3
        t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
        emb = self.time_embed(t_emb).unsqueeze(1).expand(x.size(0), x.size(1), -1).contiguous()
        x = torch.concat([x, emb], dim=-1)  # Concatenate time embedding
        x = self.to_in(x)
        
        skip_inputs = []
        skip_coords = []
        for layer in self.down_blocks:
            if isinstance(layer, PointNetSAModule):
                x = x.transpose(1, 2).contiguous()
                coords = coords.transpose(1, 2).contiguous()
                skip_inputs.append(x)
                skip_coords.append(coords)
                x, coords, _ = layer(x, coords)
                x = x.transpose(1, 2).contiguous()
                coords = coords.transpose(1, 2).contiguous()
            else:
                x = layer(x, coords=coords, context=context)
        
        x = self.bottleneck(x, coords=coords, context=context)

        for layer in self.up_blocks:
            if isinstance(layer, PointNetFPModule):
                x = x.transpose(1, 2).contiguous()
                coords = coords.transpose(1, 2).contiguous()
                x, coords, _ = layer(x, coords, skip_inputs.pop(), skip_coords.pop())
                x = x.transpose(1, 2).contiguous()
                coords = coords.transpose(1, 2).contiguous()
            else:
                x = layer(x, coords=coords, context=context)

        return self.to_out(x)

class PVCNN2Unet(nn.Module):
    DEFAULT_SA_BLOCKS = [ # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 1, 16), (256, 0.2, 32, (64, 128))),
        ((128, 1, 8), (64, 0.4, 32, (128, 128))),
        # (None, (16, 0.8, 32, (128, 128, 128))), 
    ]
    DEFAULT_FP_BLOCKS = [
        # ((128, 128), (128, 1, 8)), # fp_configs, conv_configs
        ((128, 128), (128, 1, 8)),
        ((128, 128), (128, 1, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]
    def __init__(self, 
                 use_att=True, dropout=0.1,
                 emb_dim=None,
                 extra_feature_channels=0, 
                 input_dim=3,
                 width_multiplier=1, 
                 voxel_resolution_multiplier=1,
                 sa_blocks=None, fp_blocks=None, 
                 context_dim=None
                 ):
        super().__init__()
        logger.info('[Build Unet] extra_feature_channels={}, input_dim={}',
                extra_feature_channels, input_dim)
        self.input_dim = input_dim 

        self.sa_blocks = sa_blocks if sa_blocks is not None else self.DEFAULT_SA_BLOCKS
        self.fp_blocks = fp_blocks if fp_blocks is not None else self.DEFAULT_FP_BLOCKS
        assert extra_feature_channels >= 0
        extra_feature_channels = emb_dim * 2
        self.emb_dim = emb_dim
        if self.emb_dim is not None: # has time embedding 
            self.embedf = nn.Sequential(
                nn.Linear(emb_dim, 2*emb_dim),
                nn.SiLU(),
                nn.Linear(2*emb_dim, 2*emb_dim),
            )

        self.in_channels = extra_feature_channels + self.input_dim

        sa_layers, sa_in_channels, channels_sa_features, _ = \
            create_pointnet2_sa_components(
            input_dim=input_dim,
            sa_blocks=self.sa_blocks, 
            extra_feature_channels=extra_feature_channels, 
            with_se=True,
            force_att=True, 
            emb_dim=None, # time embedding dim 
            context_dim=context_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, 
            voxel_resolution_multiplier=voxel_resolution_multiplier, 
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        if use_att:
            self.global_att = SpatialTransformerBlock(channels_sa_features, n_heads=8, dim_head=32) if context_dim is None else CondTransformerBlock(channels_sa_features, n_heads=8, dim_head=32, context_dim=context_dim)
        else:
            self.global_att = None

        # only use extra features in the last fp module
        # sa_in_channels[0] = extra_feature_channels + input_dim - 3
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, 
            sa_in_channels=sa_in_channels, force_att=True,
            with_se=True, emb_dim=None, context_dim=context_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        layers, _ = create_mlp_components(
                in_channels=channels_fp_features, 
                out_channels=[128, dropout, input_dim], # was 0.5
                classifier=True, dim=2, width_multiplier=width_multiplier,
                emb_dim=None)
        self.classifier = nn.ModuleList(layers)

    def forward(self, x, temb=None, context=None):
        x = x.transpose(1, 2).contiguous()
        # Input: coords: B3N 
        coords = x[:, :self.input_dim, :].contiguous() 
        features = x

        if temb is not None:
            temb = timestep_embedding(temb, self.emb_dim).squeeze()
            temb = self.embedf(temb)
            tembc = temb.unsqueeze(-1).expand(features.size(0), -1, features.size(2)).contiguous()
            features = torch.cat([features, tembc], dim=1)
        
        coords_list, in_features_list = [], []
        for sa_blocks in self.sa_layers:
            in_features_list.append(features)
            coords_list.append(coords)
            features, coords, _, _ = sa_blocks(features, coords, None, context) 

        # in_features_list[0] = x[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features, context=context)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            # tembc = temb.unsqueeze(-1).expand(features.size(0), -1, features.size(2)).contiguous()
            features, coords, _, _ = fp_blocks(features, coords, None, context, point_coords=coords_list[-1-fp_idx], point_feats=in_features_list[-1-fp_idx])

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                features = l(features, None)
            else:
                features = l(features)
        return features.transpose(1, 2).contiguous()