import torch
import torch.nn as nn
from collections import OrderedDict
import math
from einops import repeat

from .attention import LocalAttnBlock, PriorTransformerBlock
from .pvcnn import PointNetSAModule, PointNetFPModule

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class UNetFlow(nn.Module):
    def __init__(self,
                 depth, widths, input_dim, e_dim,
                 t_emb_ch, patch_size, num_centers,
                 n_heads, num_neighbors, radiuses, use_xformers):
        super().__init__()
        self.widths = widths
        self.input_dim = input_dim
        self.e_dim = e_dim
        self.t_emb_ch = t_emb_ch
        self.patch_size = patch_size
        self.num_centers = num_centers
        self.n_heads = n_heads
        self.num_neighbors = num_neighbors
        self.use_xformers = use_xformers
        self.radiuses = radiuses

        assert len(num_centers) == len(num_neighbors), "Num_centers length must match num_neighbors length"
        assert len(n_heads) == len(num_centers) + 1, "n_heads length must be one more than num_centers length"
        assert len(widths) == len(num_centers) + 1, "widths length must be one more than num_centers length"
        assert len(widths) == len(n_heads), "widths length must match n_heads length"
        assert len(widths) == len(radiuses) + 1, "widths length must be one more than radiuses length"

        depth = len(num_centers)

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        self.to_in = nn.Linear(self.input_dim + t_emb_dim, self.widths[0])

        layers = []

        dim_heads = [width // n_heads for width, n_heads in zip(widths, self.n_heads)]

        for i in range(depth):
            layers.append(LocalAttnBlock(
                dim=widths[i],
                context_dim=self.e_dim,
                dim_head=dim_heads[i],
                n_heads=self.n_heads[i],
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=self.patch_size,
            ))
            layers.append(PointNetSAModule(
                num_centers=num_centers[i],
                radius=radiuses[i],
                num_neighbors=num_neighbors[i],
                in_channels=widths[i],
                out_channels=widths[i+1],
            ))
        self.down_blocks = nn.Sequential(*layers)

        self.bottleneck = LocalAttnBlock(
                dim=widths[-1],
                context_dim=self.e_dim,
                dim_head=dim_heads[-1],
                n_heads=n_heads[-1],
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=self.patch_size,
            )

        layers = []

        for i in range(depth):
            layers.append(PointNetFPModule(
                in_channels=widths[-1-i] + widths[-2-i],
                out_channels=widths[-2-i],
            ))
            layers.append(LocalAttnBlock(
                dim=widths[-2-i],
                context_dim=self.e_dim,
                dim_head=dim_heads[-2-i],
                n_heads=n_heads[-2-i],
                use_xformers=self.use_xformers,
                use_pos_emb=True,
                patch_size=self.patch_size,
            ))
        self.up_blocks = nn.Sequential(*layers)

        self.to_out = nn.Linear(widths[0], self.input_dim)

    def forward(self, x, t, context):
        coords = x[:, :, :3]  # B N 3
        t_emb = timestep_embedding(t, self.t_emb_ch)
        emb = self.time_embed(t_emb).unsqueeze(1).expand(x.size(0), x.size(1), -1)
        x = torch.concat([x, emb], dim=-1)  # Concatenate time embedding
        x = self.to_in(x)
        
        skip_inputs = []
        skip_coords = []
        for layer in self.down_blocks:
            if isinstance(layer, PointNetSAModule):
                x = x.transpose(1, 2)
                coords = coords.transpose(1, 2)
                skip_inputs.append(x)
                skip_coords.append(coords)
                x, coords, _ = layer(x, coords)
                x = x.transpose(1, 2)
                coords = coords.transpose(1, 2)
            else:
                x = layer(x, coords=coords, context=context)
        
        x = self.bottleneck(x, coords=coords, context=context)

        for layer in self.up_blocks:
            if isinstance(layer, PointNetFPModule):
                x = x.transpose(1, 2)
                coords = coords.transpose(1, 2)
                x, coords, _ = layer(x, coords, skip_inputs.pop(), skip_coords.pop())
                x = x.transpose(1, 2)
                coords = coords.transpose(1, 2)
            else:
                x = layer(x, coords=coords, context=context)

        return self.to_out(x)
    
# class PriorFlow(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.depth = cfg.flow.depth
#         self.width = cfg.flow.width
#         self.input_dim = cfg.input_dim
#         self.e_dim = cfg.softvq.e_dim
#         self.latent_dim = cfg.latent_dim
#         self.t_emb_ch = cfg.flow.t_emb_ch
#         self.strides = (4, 2, 2)
#         self.channel_mult = 2
#         self.dim_head = 64
#         self.n_heads = 4
#         self.use_xformers = True
#         radius = 0.1
#         seq_len = 2048
#         num_neighbors = (32, 32, 64)

#         t_emb_dim = self.t_emb_ch * 2

#         self.time_embed = nn.Sequential(
#             nn.Linear(self.t_emb_ch, t_emb_dim),
#             nn.SiLU(),
#             nn.Linear(t_emb_dim, t_emb_dim),
#         )
#         self.to_in = nn.Linear(self.input_dim + t_emb_dim, self.width)

#         layers = []

#         widths = [self.width * (self.channel_mult ** i) for i in range(self.depth)]
#         radiuses = [radius * (i + 1) for i in range(self.depth)]

#         for width, radius, num_neighbor, stride in zip(widths, radiuses, num_neighbors, self.strides):
#             layers.append(LocalAttnBlock(
#                 dim=width,
#                 context_dim=self.e_dim,
#                 dim_head=self.dim_head,
#                 n_heads=self.n_heads,
#                 use_xformers=self.use_xformers,
#                 use_pos_emb=True,
#                 patch_size=16,
#             ))
#             layers.append(PointNetSAModule(
#                 num_centers=seq_len // stride,
#                 radius=radius,
#                 num_neighbors=num_neighbor,
#                 in_channels=width,
#                 out_channels=width * self.channel_mult,
#             ))
#         self.down_blocks = nn.Sequential(*layers)

#         self.bottleneck = LocalAttnBlock(
#                 dim=width * self.channel_mult,
#                 context_dim=self.e_dim,
#                 dim_head=self.dim_head,
#                 n_heads=self.n_heads,
#                 use_xformers=self.use_xformers,
#                 use_pos_emb=True,
#                 patch_size=16,
#             )

#         layers = []

#         for width in reversed(widths):
#             layers.append(PointNetFPModule(
#                 in_channels=width * self.channel_mult + width,
#                 out_channels=width,
#             ))
#             layers.append(LocalAttnBlock(
#                 dim=width,
#                 context_dim=self.e_dim,
#                 dim_head=self.dim_head,
#                 n_heads=self.n_heads,
#                 use_xformers=self.use_xformers,
#                 use_pos_emb=True,
#                 patch_size=16,
#             ))
#         self.up_blocks = nn.Sequential(*layers)

#         self.to_out = nn.Linear(self.width, self.input_dim)

#     def forward(self, x, t):
#         coords = x[:, :, :3].contiguous()  # B N 3
#         t_emb = timestep_embedding(t, self.t_emb_ch).squeeze()
#         emb = self.time_embed(t_emb).unsqueeze(1).expand(x.size(0), x.size(1), -1).contiguous()
#         x = torch.concat([x, emb], dim=-1)  # Concatenate time embedding
#         x = self.to_in(x)
        
#         skip_inputs = []
#         skip_coords = []
#         for layer in self.down_blocks:
#             if isinstance(layer, PointNetSAModule):
#                 x = x.transpose(1, 2).contiguous()
#                 coords = coords.transpose(1, 2).contiguous()
#                 skip_inputs.append(x)
#                 skip_coords.append(coords)
#                 x, coords, _ = layer(x, coords)
#                 x = x.transpose(1, 2).contiguous()
#                 coords = coords.transpose(1, 2).contiguous()
#             else:
#                 x = layer(x, coords=coords)
        
#         x = self.bottleneck(x, coords=coords)

#         for layer in self.up_blocks:
#             if isinstance(layer, PointNetFPModule):
#                 x = x.transpose(1, 2).contiguous()
#                 coords = coords.transpose(1, 2).contiguous()
#                 x, coords, _ = layer(x, coords, skip_inputs.pop(), skip_coords.pop())
#                 x = x.transpose(1, 2).contiguous()
#                 coords = coords.transpose(1, 2).contiguous()
#             else:
#                 x = layer(x, coords=coords)

#         return self.to_out(x)


class PriorFlow(nn.Module):
    """
    Transformer-based flow model for the latent prior.
    Operates on abstract token sequences.
    """
    def __init__(self, cfg):
        super().__init__()
        # All config from cfg.prior — no cfg.vae keys needed here
        self.depth = cfg.prior.depth
        self.width = cfg.prior.width
        self.e_dim = cfg.vae.softvq.e_dim       # token dimension = input/output dim
        self.t_emb_ch = cfg.prior.t_emb_ch
        self.n_heads = cfg.prior.n_heads
        self.dim_head = cfg.prior.dim_head
        self.use_xformers = cfg.prior.use_xformers

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.t_emb_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        # Project (e_dim + time_emb) -> width
        self.to_in = nn.Linear(self.e_dim + t_emb_dim, self.width)

        self.blocks = nn.ModuleList([
            PriorTransformerBlock(
                dim=self.width,
                n_heads=self.n_heads,
                dim_head=self.dim_head,
                use_xformers=self.use_xformers,
            )
            for _ in range(self.depth)
        ])

        # Project back to token dimension
        self.to_out = nn.Linear(self.width, self.e_dim)

    def forward(self, x, t):
        """
        Args:
            x: noisy latent tokens (B, seq_len, e_dim)
            t: time (B,) or (B, 1)
        Returns:
            velocity field (B, seq_len, e_dim)
        """
        t = t.view(t.shape[0])  # ensure (B,)
        t_emb = timestep_embedding(t, self.t_emb_ch)               # (B, t_emb_ch)
        emb = self.time_embed(t_emb).unsqueeze(1).expand(-1, x.size(1), -1)  # (B, N, t_emb_dim)

        x = torch.cat([x, emb], dim=-1)   # (B, N, e_dim + t_emb_dim)
        x = self.to_in(x)                 # (B, N, width)

        for block in self.blocks:
            x = block(x)

        return self.to_out(x)             # (B, N, e_dim)