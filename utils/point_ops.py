from pointnet2_ops import pointnet2_utils
import torch

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist 


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx


def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device) # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out) # (M, D/6)
        emb_cos = torch.cos(out) # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)

def get_pos_embed(embed_dim, ipt_pos):
    """
    embed_dim: output dimension for each position
    ipt_pos: [B, G, 3], where 3 is (x, y, z)
    """
    B, G, _ = ipt_pos.size()
    assert embed_dim % 6 == 0
    omega = torch.arange(embed_dim // 6).float().to(ipt_pos.device) # NOTE
    omega /= embed_dim / 6.
    # (0-31) / 32
    omega = 1. / 10000**omega  # (D/6,)
    rpe = []
    for i in range(_):
        pos_i = ipt_pos[:, :, i]    # (B, G)
        out = torch.einsum('bg, d->bgd', pos_i, omega)  # (B, G, D/6), outer product
        emb_sin = torch.sin(out) # (M, D/6)
        emb_cos = torch.cos(out) # (M, D/6)
        rpe.append(emb_sin)
        rpe.append(emb_cos)
    return torch.cat(rpe, dim=-1)