class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, include_coordinates=True, cfg={}):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = 1  # Assuming 1D convolution for PointNet
        self.shared_mlp = SharedMLP(in_channels, out_channels, dim=self.dim, cfg=cfg)

    def forward(self, x):
        return self.shared_mlp(x)