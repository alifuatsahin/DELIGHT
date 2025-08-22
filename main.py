# from models import VAE, DDPM
# from default_config import cfg

from geomloss import SamplesLoss
from pykeops.torch import LazyTensor
from geomloss.utils import squared_distances
from utils.eval_metrics import distChamferCUDAnograd
import os

import time
import torch

if __name__ == "__main__":
    pc = torch.randn(32, 2048, 3)  # Example point cloud
    pc2 = torch.randn(32, 2048, 3)

    loss = SamplesLoss(blur=0.05, p=2, potentials=True, backend="online")
    start_time = time.time()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        F, G = loss(pc, pc2)
        print("F shape:", F.shape)
        print("G shape:", G.shape)

        epsilon = 0.05 ** 2

        # Encoding as batched KeOps LazyTensors:
        x_i = LazyTensor(pc[:, :, None, :])  # (B, N, 1, D)
        y_j = LazyTensor(pc2[:, None, :, :])  # (B, 1, M, D)

        # Cost matrix:
        C_ij = ((x_i - y_j) ** 2).sum(-1) / 2  # (B, N, M, 1)
        F_i = LazyTensor(F[:, :, None, None])  # (B, N, 1, 1)
        G_j = LazyTensor(G[:, None, :, None])  # (B, 1, M, 1)

        T = ((F_i + G_j - C_ij) / epsilon).exp() # (B, N, M, 1)

        print(f"x1_j shape: {y_j.shape}")

        num = (T * y_j).sum(dim=2)  # (B, N, D)

        print(f"T shape: {T.shape}")

        print(f"num shape: {num.shape}")

        den = T.sum(dim=2)  # (B, N, 1)

        print(f"den shape: {den.shape}")

        y_hat = (num / den.clamp_min(1e-12))

        print("y_hat shape:", y_hat.shape)

        dl, dr = distChamferCUDAnograd(y_hat, pc2)

        print("Chamfer distance:", (dl.mean(dim=1) + dr.mean(dim=1)).mean())

    print(prof.key_averages().table(sort_by="cuda_time_total"))
    # print(T)
    # print(matched_pc2.shape)
    # print(type(matched_pc2))
    print("Time taken:", time.time() - start_time)

    # grid_size = 0.01
    # depth = 16
    # order = "z"

    # # Encode
    # code, order_pc, inverse = encode(pc, grid_size=grid_size, depth=depth, order=order)
    # print("Encoded max index:", order_pc.max(1))
    # print("Encoded min index:", order_pc.min(1))

    # # Decode
    # decoded_pc, batch = decode(code.view(-1), depth=depth, order=order)
    # print("Decoded successfully")

    # # Recompute quantized grid coordinates for comparison
    # grid_coord = torch.div(
    #     pc - pc.min(1, keepdim=True)[0], 0.01, rounding_mode="trunc"
    # ).int().view(-1, 3)

    # # Compare decoded_grid to grid_coord
    # decoded_flat = decoded_pc.view(-1, 3)
    # assert torch.all(decoded_flat == grid_coord), "Decoded grid does not match quantized input!"

    # order_pc_expanded = order_pc.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    # inverse_pc_expanded = inverse.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    # # Order pointcloud
    # pc_ordered = torch.gather(pc, 1, order_pc_expanded)

    # # Inverse it and check if its correct
    # pc_reverted = torch.gather(pc_ordered, 1, inverse_pc_expanded)
    # assert torch.all(pc_reverted == pc), "Inverse ordering does not match original point cloud!"

    # print("Round-trip test passed!")

    # assert torch.unique(code).numel() == code.numel(), "Codes are not unique!"
    # print("Uniqueness test passed!")