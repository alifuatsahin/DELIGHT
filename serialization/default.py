import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


def encode(pc, grid_size=0.01, depth=16, order="z"):
    '''
    Encode point cloud coordinates into a compact representation.
    Inputs:
      input (torch.Tensor): Point clouds (B, C, N)
      grid_size (float): The size of the grid for voxelization.
      depth (int): The depth of the encoding.
      order (str): The order of the encoding.
    '''
    B, N, C = pc.shape
    pc = pc[:, :, :3]  # Ensure only the point coordinates are used
    grid_coord = torch.div(
        pc - pc.min(1, keepdim=True)[0], grid_size, rounding_mode="trunc"
    ).int().view(-1, 3)

    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError

    # Add batch offset for uniqueness
    batch = torch.arange(B, device=pc.device).repeat_interleave(N).long()
    code = batch << depth * 3 | code
    code = code.view(B, N)
    order = torch.argsort(code, dim=1)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(N, device=order.device).expand(B, N),
        )
    return code, order, inverse


def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z, _ = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (B, N, 3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)