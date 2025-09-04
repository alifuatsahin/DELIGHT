import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from einops import repeat

from .attention import TransformerBlock
from .layers import CondResBlock, SpatialMLP
from serialization import encode

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

class CondSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings or context to the children that
    support it as an extra input.
    """

    def forward(self, x, context):
        for layer in self:
            if isinstance(layer, CondResBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out * 2, kernel_size=1)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = nn.Sequential(
            nn.Conv1d(dim, inner_dim, kernel_size=1),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim, dim_out, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dim_head, dropout=0., use_xformers=True, context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = TransformerBlock(dim, n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=True)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = TransformerBlock(dim, context_dim=context_dim,
                                    n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=True)  # is self-attn if context is none
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.norm3 = nn.GroupNorm(1, dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
    def __init__(self, channels, out_channels, n_heads, dim_head,
                 depth=1, dropout=0., context_dim=None, use_xformers=True):
        super().__init__()
        self.in_channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        inner_dim = n_heads * dim_head
        self.norm = nn.GroupNorm(1, channels)

        self.proj_in = nn.Conv1d(channels,
                                 inner_dim,
                                 kernel_size=1)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, dim_head, dropout=dropout, context_dim=context_dim, use_xformers=use_xformers)
                for _ in range(depth)]
        )

        self.proj_out = nn.Conv1d(inner_dim,
                                  self.out_channels,
                                  kernel_size=1)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj_out(x)
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
            # layers.append(SpatialMLP(
            #     channels=self.width,
            #     dropout=0.0,
            #     out_channels=self.width,
            # ))
            layers.append(CondResBlock(
                channels=self.width,
                emb_channels=t_emb_dim,
                dropout=0.0,
                out_channels=self.width,
                use_scale_shift_norm=True
            ))
            layers.append(TransformerBlock(
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