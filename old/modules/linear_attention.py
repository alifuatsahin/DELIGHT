import torch
import torch.nn as nn
from einops import rearrange

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super.__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, 1, bias=True)

    def forward(self, x):
        '''
        Args:
            x: (B, N, C) tensor where B is batch size, N is num points, and C is feature dimension.
        Returns:
            out: (B, N, C) tensor after applying linear attention.    
        '''

        x = x.unsqueeze(-1) # (B, N, C, 1)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  # Normalize along the last dimension
        context = torch.einsum('bhdn,bhen->bhde', k, v)  # (B, heads, C, N)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        out = self.to_out(out)
        return out.squeeze(-1)
