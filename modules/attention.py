import torch
import torch.nn as nn
from einops import rearrange
import xformers.ops as xops
import math


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
    
class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", 
            (q * scale).reshape(B * self.n_heads, ch, H * W),
            (k * scale).reshape(B * self.n_heads, ch, H * W),
        ) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(B * self.n_heads, ch, H * W))
        return a.reshape(B, -1, H * W)

class EfficientQKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v):
        B, C, H, W = q.shape
        assert C % self.n_heads == 0
        ch = C // self.n_heads
        q = q.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()  # [B, H*W, n_heads, ch]
        k = k.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, self.n_heads, ch, H * W).permute(0, 3, 1, 2).contiguous()
        scale = 1 / math.sqrt(math.sqrt(ch))
        y = xops.memory_efficient_attention(q, k, v, scale=scale)
        y = y.permute(0, 2, 3, 1).reshape(B, self.n_heads * ch, H * W).unsqueeze(-1)
        return y

class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=4, hidden_dim=4*32, use_xformers=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert dim % n_heads == 0, "in_channels must be divisible by n_heads"
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1)
        if use_xformers:
            self.attn = EfficientQKVAttention(n_heads)
        else:
            self.attn = QKVAttention(n_heads)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # x: (B, C, N)
        x = x.unsqueeze(-1) # add w dimensio
        qkv = self.to_qkv(x)  # (B, 3*C, N)
        q, k, v = qkv.chunk(3, dim=1)  # (B, C, N)
        out = self.attn(q, k, v)  # (B, C, N)
        out = self.to_out(out)
        out = out.squeeze(-1)  # B, C, N, 1 -> B, C, N
        return out

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, n_heads=4, dim_head=64, dropout=0., use_xformers=True):
        super().__init__()
        inner_dim = dim_head * n_heads
        if context_dim is None:
            context_dim = query_dim
        self.heads = n_heads

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1)
        self.to_kv = nn.Conv2d(context_dim, 2 * inner_dim, 1)

        if use_xformers:
            self.attn = EfficientQKVAttention(n_heads)
        else:
            self.attn = QKVAttention(n_heads)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # x: (B, C, N)
        x = x.unsqueeze(-1) # add w dimension
        q = self.to_q(x)
        if context is None:
            context = x
        else:
            context = context.unsqueeze(-1)  # add w dimension
        kv = self.to_kv(context)
        k, v = kv.chunk(2, dim=1)
        out = self.attn(q, k, v)  # (B, C, N)
        out = self.to_out(out)
        out = out.squeeze(-1)  # B, C, N, 1 ->
        return out