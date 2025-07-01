import torch.nn as nn
from torch.nn import Sequential
from modules.swish import Swish

class SE3d(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self,fc = nn,Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.channel = channel
    def __repr__(self):
        return f"SE({self.channel}, {self.channel})"

    def forward(self, x):
        return x * self.fc(x.mean(-1).mean(-1).mean(-1)).view(x.shape[0], x.shape[1], 1, 1, 1)