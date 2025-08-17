# from models import VAE, DDPM
# from default_config import cfg

from serialization import encode, decode
from modules.flows import timestep_embedding

import torch

if __name__ == "__main__":
    pc = torch.randn(10, 1024, 3)  # Example point cloud
    grid_size = 0.01
    depth = 16
    order = "z"

    # Encode
    code, order_pc, inverse = encode(pc, grid_size=grid_size, depth=depth, order=order)
    print("Encoded max index:", order_pc.max(1))
    print("Encoded min index:", order_pc.min(1))

    # Decode
    decoded_pc, batch = decode(code.view(-1), depth=depth, order=order)
    print("Decoded successfully")

    # Recompute quantized grid coordinates for comparison
    grid_coord = torch.div(
        pc - pc.min(1, keepdim=True)[0], 0.01, rounding_mode="trunc"
    ).int().view(-1, 3)

    # Compare decoded_grid to grid_coord
    decoded_flat = decoded_pc.view(-1, 3)
    assert torch.all(decoded_flat == grid_coord), "Decoded grid does not match quantized input!"

    order_pc_expanded = order_pc.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    inverse_pc_expanded = inverse.unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    # Order pointcloud
    pc_ordered = torch.gather(pc, 1, order_pc_expanded)

    # Inverse it and check if its correct
    pc_reverted = torch.gather(pc_ordered, 1, inverse_pc_expanded)
    assert torch.all(pc_reverted == pc), "Inverse ordering does not match original point cloud!"

    print("Round-trip test passed!")

    assert torch.unique(code).numel() == code.numel(), "Codes are not unique!"
    print("Uniqueness test passed!")