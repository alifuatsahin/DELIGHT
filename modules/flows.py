import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from einops import repeat

from .attention import TransformerBlock
from .layers import ResBlock
from utils.diffusion_helper import zero_module

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [B x dim x N] Tensor of positional embeddings.
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

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
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
        self.attn1 = TransformerBlock(dim, n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = TransformerBlock(dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=dim_head, use_xformers=use_xformers)  # is self-attn if context is none
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

class FiLMCond(nn.Module):
    def __init__(self, input_dim, out_dim, context_dim, weight_std=0.001, bias=0.0, *args, **kwargs):
        super().__init__()
        self.layer = nn.Sequential([
            nn.Conv1d(input_dim, out_dim, kernel_size=1),
        ])

        self.gate = nn.Sequential(OrderedDict([
            nn.Linear(context_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True)
        ]))

        self.bias = nn.Sequential(OrderedDict([
            nn.Linear(context_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True)
        ]))

        with torch.no_grad():
            self.layer[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.layer[-1].bias.data, bias)
            self.gate[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.gate[-1].bias.data, bias)
            self.bias[-1].weight.data.normal_(std=weight_std)
            nn.init.constant_(self.bias[-1].bias.data, bias)

    def forward(self, x, context):
        if context.dim() > 2:
            context = context.view(context.size(0), -1).contiguous()

        g = torch.add(F.softplus(self.gate(context).unsqueeze(-1)), 1e-6)  # Ensure gate is positive
        b = self.bias(context).unsqueeze(-1)
        out = g * self.layer(x) + b
        return out

class FlowAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.num_res_blocks = cfg.flow.num_res_blocks
        self.width = cfg.flow.width
        self.n_heads = cfg.flow.n_heads
        self.dim_head = cfg.flow.dim_head
        self.e_dim = cfg.softvq.e_dim
        self.input_dim = cfg.input_dim
        self.latent_dim = cfg.latent_dim
        self.t_emb_ch = cfg.flow.t_emb_ch
        self.use_xformers = cfg.flow.use_xformers_attention

        t_emb_dim = self.t_emb_ch * 2

        self.time_embed = nn.Sequential(
            nn.Conv1d(self.t_emb_ch, t_emb_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(t_emb_dim, t_emb_dim, kernel_size=1),
        )

        self.layers = SpatialTransformer(
            channels=self.input_dim + t_emb_dim,
            out_channels=self.input_dim,
            n_heads=self.n_heads,
            dim_head=self.dim_head,
            context_dim=self.e_dim,
            dropout=0.0,
            use_xformers=self.use_xformers,
            depth=self.depth,
        )

    def forward(self, x, t, context):
        t_emb = timestep_embedding(t, self.t_emb_ch)
        emb = self.time_embed(t_emb)
        h = torch.cat([x, emb], dim=1)  # Concatenate time embedding
        h = self.layers(h, context)
        return h

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
            layers.append(ResBlock(
                channels=self.width,
                emb_channels=self.latent_dim,
                dropout=0.0,
                out_channels=self.width,
                use_scale_shift_norm=True
            ))
        self.hidden_blocks = CondSequential(*layers)

        self.to_out = nn.Conv1d(self.width, self.input_dim, kernel_size=1)

    def forward(self, x, t, context):
        context = context.view(context.size(0), -1).contiguous() if context.dim() > 2 else context
        t_emb = timestep_embedding(t, self.t_emb_ch)
        emb = self.time_embed(t_emb)
        h = torch.cat([x, emb], dim=1)  # Concatenate time embedding
        h = self.to_in(h)
        h = self.hidden_blocks(h, context)
        return self.to_out(h)

class FlowModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.depth = cfg.flow.depth
        self.num_res_blocks = cfg.flow.num_res_blocks
        self.width = cfg.flow.width
        self.e_dim = cfg.softvq.e_dim
        self.input_dim = cfg.input_dim

        t_emb_dim = self.width * 2

        self.time_embed = nn.Sequential(
            nn.Linear(self.width, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.to_in = nn.Sequential(
            zero_module(nn.Conv1d(self.input_dim, self.width, 1)),
        )

        self.input_blocks = nn.ModuleList([])
        self.output_blocks = nn.ModuleList([])

        layers = []
        layers.append(ResBlock(
            channels=self.width,
            emb_channels=t_emb_dim,
            dropout=0.0,
            out_channels=self.width,
            use_scale_shift_norm=True
        ))
        layers.append(SpatialTransformer(
            dim=self.width,
            context_dim=self.e_dim,
            n_heads=cfg.flow.n_heads,
            dim_head=cfg.flow.dim_head,
            depth=cfg.flow.attn_depth,
            use_xformers=cfg.flow.use_xformers_attention,
        ))
        layers.append(ResBlock(
            channels=self.width,
            emb_channels=t_emb_dim,
            dropout=0.0,
            out_channels=self.width,
            use_scale_shift_norm=True
        ))

        self.middle_blocks = CondSequential(*layers)

        for _ in range(self.depth):
            layers = []
            for _ in range(self.num_res_blocks):
                layers.append(ResBlock(
                    channels=self.width,
                    emb_channels=t_emb_dim,
                    dropout=0.0,
                    out_channels=self.width,
                    use_scale_shift_norm=True
                ))
            layers.append(SpatialTransformer(
                dim=self.width,
                context_dim=self.e_dim,
                n_heads=cfg.flow.n_heads,
                dim_head=cfg.flow.dim_head,
                depth=cfg.flow.attn_depth,
                use_xformers=cfg.flow.use_xformers_attention,
            ))
            self.input_blocks.append(CondSequential(*layers))
            for _ in range(self.num_res_blocks):
                layers.append(ResBlock(
                    channels=self.width,
                    emb_channels=t_emb_dim,
                    dropout=0.0,
                    out_channels=self.width,
                    use_scale_shift_norm=True
                ))
            layers.append(SpatialTransformer(
                dim=self.width,
                context_dim=self.e_dim,
                n_heads=cfg.flow.n_heads,
                dim_head=cfg.flow.dim_head,
                depth=cfg.flow.attn_depth,
                use_xformers=cfg.flow.use_xformers_attention,
            ))
            self.output_blocks.append(CondSequential(*layers))

        self.to_out = nn.Sequential(
            nn.BatchNorm1d(self.width),
            nn.SiLU(),
            zero_module(nn.Conv1d(self.width, cfg.input_dim, 1)),
        )

    def forward(self, x, t, context):
        hs = []
        t_emb = timestep_embedding(t, self.width).transpose(2, 1)  # (B, D, C)
        emb = self.time_embed(t_emb)
        x = self.to_in(x)
        for module in self.input_blocks:
            x = module(x, emb, context)
            # hs.append(x)
        x = self.middle_blocks(x, emb, context)
        for module in self.output_blocks:
            # x = torch.cat([x, hs.pop()], dim=1)
            x = module(x, emb, context)
        return self.to_out(x)