import torch
import torch.nn as nn

class CondResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, self.out_channels),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.out_channels, self.out_channels),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(channels, self.out_channels)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class SpatialCondResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, 1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.BatchNorm1d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(self.out_channels, self.out_channels, 1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
class SpatialResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout=0.0,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.layers = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, self.out_channels, kernel_size=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        h = self.layers(x)
        return self.skip_connection(x) + h

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout=0.0,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(channels, self.out_channels),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(channels, self.out_channels)

    def forward(self, x):
        h = self.layers(x)
        return self.skip_connection(x) + h