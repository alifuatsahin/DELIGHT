import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import xformers.ops as xops
import math

from serialization import encode

class SpatialFixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=2, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i, j -> j i', t, self.inv_freq)
        emb = torch.zeros(self.inv_freq.shape[0] * 2, x.shape[seq_dim], device=x.device, dtype=sinusoid_inp.dtype)
        emb[0::2, :] = sinusoid_inp.sin()
        emb[1::2, :] = sinusoid_inp.cos()
        return emb[None, :, :]
    
class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]

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
    
class SpatialQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, C, N = q.shape
        _, _, M = k.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        scale = 1 / math.sqrt(ch)
        weight = torch.einsum(
            "bct,bcs->bts", 
            q.reshape(B * self.n_heads, ch, N),
            k.reshape(B * self.n_heads, ch, M),
        ) * scale # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(B * self.n_heads, ch, M))
        return a.reshape(B, -1, N)

class EfficientSpatialQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, C, N = q.shape
        _, _, M = k.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        q = q.reshape(B, self.n_heads, ch, N).permute(0, 3, 1, 2).contiguous()  # [B, N, n_heads, ch]
        k = k.reshape(B, self.n_heads, ch, M).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, self.n_heads, ch, M).permute(0, 3, 1, 2).contiguous()
        scale = 1 / math.sqrt(ch)
        y = xops.memory_efficient_attention(q, k, v, scale=scale)
        y = y.permute(0, 2, 3, 1).reshape(B, self.n_heads * ch, N)
        return y

class SpatialTransformerBlock(nn.Module):
    def __init__(self, dim, context_dim=None, n_heads=4, dim_head=64, use_xformers=True, use_pos_emb=False):
        super().__init__()
        inner_dim = dim_head * n_heads
        if context_dim is None:
            context_dim = dim
        self.heads = n_heads

        self.pos_emb = SpatialFixedPositionalEmbedding(dim) if use_pos_emb else None
        self.c_pos_emb = SpatialFixedPositionalEmbedding(context_dim) if use_pos_emb and context_dim else None

        self.to_q = nn.Conv1d(dim, inner_dim, 1)
        self.to_kv = nn.Conv1d(context_dim, 2 * inner_dim, 1)

        if use_xformers:
            self.attn = EfficientSpatialQKVAttention(n_heads)
        else:
            self.attn = SpatialQKVAttention(n_heads)

        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x, context=None):
        # x: (B, C, N)
        pos_emb = self.pos_emb(x) if self.pos_emb else 0
        x = x + pos_emb
        q = self.to_q(x)
        if context is None:
            context = x
        else:
            c_pos_emb = self.c_pos_emb(context) if self.c_pos_emb else 0
            context = context + c_pos_emb
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=1)
        out = self.attn(q, k, v)  # (B, C, N)
        out = self.to_out(out)
        return out

class LocalTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dim_head=64, use_pos_emb=True, patch_size=256):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.n_heads = n_heads
        self.patch_size = patch_size
        
        self.rpe = RPE(patch_size, n_heads) if use_pos_emb else None

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(dim, 2 * inner_dim)

        self.to_out = nn.Linear(inner_dim, dim)
        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

    @torch.no_grad()
    def serialize(self, coords, grid_size=0.01, depth=16):
        idx = torch.randint(len(self.orders), (1,)).item()
        _, order, inverse = encode(coords, grid_size=grid_size, depth=depth, order=self.orders[idx])
        return order, inverse

    @torch.no_grad()
    def get_rel_pos(self, coords):
        K = self.patch_size
        grid_coord = torch.div(
            coords - coords.min(1, keepdim=True)[0], 0.01, rounding_mode="trunc"
        ).int().view(-1, 3)
        grid_coord = grid_coord.reshape(-1, K, 3)
        rel_pos = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1) # (B * n_patches, K, K, 3)
        return rel_pos

    def patchify(self, x, coords):
        B, N, C = x.shape
        # Calculate number of patches per batch
        num_patches = (N + self.patch_size - 1) // self.patch_size  # ceil division

        # Pad so each batch has a multiple of patch_size points
        pad_N = num_patches * self.patch_size - N
        if pad_N > 0:
            pad = x[:, -pad_N:, :]  # repeat last points
            pad_coords = coords[:, -pad_N:, :]
            x_padded = torch.cat([x, pad], dim=1)
            coords_padded = torch.cat([coords, pad_coords], dim=1)
        else:
            x_padded = x
            coords_padded = coords

        # Reshape into patches
        x_patches = x_padded.view(B * num_patches, self.patch_size, C)  # (B * num_patches, patch_size, C)
        coords_padded = coords_padded.view(B * num_patches, self.patch_size, 3)
        return x_patches, coords_padded

    def attn(self, q, k, v, rel_pos=0):
        B, N, C = q.shape
        _, M, C = k.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        scale = 1 / math.sqrt(ch)

        # reshape into heads
        q = q.view(B, N, self.n_heads, ch).permute(0, 2, 1, 3)  # [B, H, N, ch]
        k = k.view(B, M, self.n_heads, ch).permute(0, 2, 1, 3)  # [B, H, M, ch]
        v = v.view(B, M, self.n_heads, ch).permute(0, 2, 1, 3)  # [B, H, M, ch]

        # attention logits: [B, H, N, M]
        weight = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale

        weight = weight + rel_pos

        weight = torch.softmax(weight, dim=-1)
        
        # weighted sum: [B, H, N, ch]
        a = torch.einsum("bhnm,bhmd->bhnd", weight, v)

        # merge heads back: [B, N, C]
        out = a.permute(0, 2, 1, 3).reshape(B, N, C)
        return out

    def forward(self, x, coords):
        B, N, C = x.shape
        order, inverse = self.serialize(coords.contiguous())  # (B, N, C)
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, C)) # (B, N, C)
        coords = torch.gather(coords, 1, order.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)
        x, coords = self.patchify(x, coords)  # (B * num_patches, patch_size, C)

        rel_pos = self.rpe(self.get_rel_pos(coords)) if self.rpe else 0

        q = self.to_q(x)  # (B * num_patches, patch_size, C)
        kv = self.to_kv(x) # (B * num_patches, patch_size, 2*C)
        k, v = kv.chunk(2, dim=-1)

        attn = self.attn(q, k, v, rel_pos=rel_pos)
        out = self.to_out(attn).view(B, -1, C)[:, :N, :]  # (B, N, C)

        return torch.gather(out, 1, inverse.unsqueeze(-1).expand(-1, -1, C))  # unshuffle

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, n_heads=4, dim_head=64, use_xformers=True, use_pos_emb=False):
        super().__init__()
        inner_dim = dim_head * n_heads
        if context_dim is None:
            context_dim = dim
        self.heads = n_heads

        self.pos_emb = FixedPositionalEmbedding(dim) if use_pos_emb else None
        self.c_pos_emb = FixedPositionalEmbedding(context_dim) if (use_pos_emb and context_dim) else None

        self.to_q = nn.Linear(dim, inner_dim)
        self.to_kv = nn.Linear(context_dim, 2 * inner_dim)

        if use_xformers:
            self.attn = EfficientQKVAttention(n_heads)
        else:
            self.attn = QKVAttention(n_heads)

        self.to_out = nn.Linear(inner_dim, dim)
        self.orders = ["z", "hilbert", "hilbert-trans", "z-trans"]

    def serialize(self, coords, grid_size=0.01, depth=16):
        idx = torch.randint(len(self.orders), (1,)).item()
        _, order, inverse = encode(coords, grid_size=grid_size, depth=depth, order=self.orders[idx])
        return order, inverse

    def forward(self, x, coords, context=None):
        # x: (B, N, C)
        order, inverse = self.serialize(coords.contiguous())  # (B, N, C)
        x = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, x.shape[-1]))  # (B, N, C)
        coords = torch.gather(coords, 1, order.unsqueeze(-1).expand(-1, -1, 3))  # (B, N, 3)
        pos_emb = self.pos_emb(x) if self.pos_emb else 0
        x = x + pos_emb
        q = self.to_q(x)
        if context is None:
            context = x
        else:
            c_pos_emb = self.c_pos_emb(context) if self.c_pos_emb else 0
            context = context + c_pos_emb
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)
        out = self.attn(q, k, v)  # (B, N, C)
        out = self.to_out(out)
        return torch.gather(out, 1, inverse.unsqueeze(-1).expand(-1, -1, out.shape[-1]))  # unshuffle

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, N, C = q.shape
        _, M, _ = k.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        scale = 1 / math.sqrt(ch)
        weight = torch.einsum(
            "bhnd,bhmd->bhnm", 
            q.reshape(B, self.n_heads, N, ch),
            k.reshape(B, self.n_heads, M, ch),
        ) * scale  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bhnm,bhmd->bhnd", weight, v.reshape(B, self.n_heads, M, ch))
        return a.permute(0, 2, 1, 3).reshape(B, N, -1)

class EfficientQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, N, C = q.shape
        _, M, _ = k.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        q = q.reshape(B, self.n_heads, N, ch).permute(0, 2, 1, 3).contiguous()  # [B, N, n_heads, ch]
        k = k.reshape(B, self.n_heads, M, ch).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, self.n_heads, M, ch).permute(0, 2, 1, 3).contiguous()
        scale = 1 / math.sqrt(ch)
        y = xops.memory_efficient_attention(q, k, v, scale=scale)
        y = y.reshape(B, N, self.n_heads * ch)
        return y

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dim_head=64, dropout=0., use_xformers=True, context_dim=None, gated_ff=True, use_pos_emb=False):
        super().__init__()
        self.attn1 = CrossAttention(dim, n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=use_pos_emb)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(dim, context_dim=context_dim,
                                    n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=use_pos_emb)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, coords, context=None):
        x = self.attn1(self.norm1(x), coords=coords) + x
        x = self.attn2(self.norm2(x), coords=coords, context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class LocalAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dim_head=64, dropout=0., patch_size=16, use_xformers=True, context_dim=None, gated_ff=True, use_pos_emb=False):
        super().__init__()
        self.attn1 = LocalTransformerBlock(dim, n_heads=n_heads, dim_head=dim_head, use_pos_emb=use_pos_emb, patch_size=patch_size)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(dim, context_dim=context_dim,
                                    n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=use_pos_emb)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, coords, context=None):
        x = self.attn1(self.norm1(x), coords=coords) + x
        x = self.attn2(self.norm2(x), coords=coords, context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialGEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out * 2, kernel_size=1)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * F.gelu(gate)

class SpatialFeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = nn.Sequential(
            nn.Conv1d(dim, inner_dim, kernel_size=1),
            nn.GELU()
        ) if not glu else SpatialGEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim, dim_out, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class SpatialAttnBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dim_head=64, dropout=0., use_xformers=True, context_dim=None, gated_ff=True, use_pos_emb=False):
        super().__init__()
        self.attn1 = SpatialTransformerBlock(dim, n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=use_pos_emb)  # is a self-attention
        self.ff = SpatialFeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = SpatialTransformerBlock(dim, context_dim=context_dim,
                                    n_heads=n_heads, dim_head=dim_head, use_xformers=use_xformers, use_pos_emb=use_pos_emb)  # is self-attn if context is none
        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.norm3 = nn.GroupNorm(1, dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out
        
class PriorTransformerBlock(nn.Module):
    """A single transformer block for the latent prior: self-attention + FFN, no geometry."""
    def __init__(self, dim, n_heads, dim_head, use_xformers=True):
        super().__init__()
        inner_dim = n_heads * dim_head
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.ff = FeedForward(dim, glu=True)
        self.attn = EfficientQKVAttention(n_heads) if use_xformers else QKVAttention(n_heads)

    def forward(self, x):
        # x: (B, N, D)
        h = self.norm1(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        x = x + self.to_out(self.attn(q, k, v))
        x = x + self.ff(self.norm2(x))
        return x