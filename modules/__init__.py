import torch.nn as nn
import functools
from .pvcnn import SharedMLP, PVConv, PointNetAModule, PointNetSAModule, PointNetFPModule, PointNetSequential

def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.GroupNorm(8, out_channels),
        nn.SiLU()
    )

def create_mlp_components(in_channels, out_channels, classifier=False, emb_dim=None, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_gn_relu
    else:
        block = functools.partial(SharedMLP, emb_dim=emb_dim)

    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels
    
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(oc * r)
            layers.append(block(in_channels, oc))
            in_channels = oc

    if dim == 1 and classifier:
        layers.append(nn.Linear(in_channels, out_channels[-1]))
    elif classifier:
        layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
    else:
        layers.append(block(in_channels, int(r * out_channels[-1])))

    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])

def create_pointnet2_sa_components(
        sa_blocks, 
        extra_feature_channels,
        input_dim=3,
        emb_dim=None,
        context_dim=None,
        use_att=False,
        force_att=0,
        dropout=0.1,
        with_se=False,
        normalize=True,
        eps=0,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + input_dim
    sa_layers, sa_in_channels = [], []
    c = 0
    num_centers = None
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)

            for p in range(num_blocks):
                attention = ((c) % 2 == 0 and use_att and p == 0) or (force_att and c > 0)
                if voxel_resolution is None:
                    block = functools.partial(SharedMLP, emb_dim=emb_dim)
                else:
                    block = functools.partial(
                        PVConv, 
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution), 
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        normalize=normalize,
                        eps=eps,
                        emb_dim=emb_dim,
                        context_dim=context_dim
                    )

                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels

            extra_feature_channels = in_channels

        if sa_configs is not None:
            num_centers, radius, num_neighbors, out_channels = sa_configs
            _out_channels = []
            for oc in out_channels:
                if isinstance(oc, (list, tuple)):
                    _out_channels.append([int(r * _oc) for _oc in oc])
                else:
                    _out_channels.append(int(r * oc))

            out_channels = _out_channels
            if num_centers is None:
                block = PointNetAModule
            else:
                block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius, num_neighbors=num_neighbors)
            
            sa_blocks.append(block(
                in_channels=extra_feature_channels,
                out_channels=out_channels,
                include_coordinates=True,
                emb_dim=emb_dim
            ))
            in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        c += 1

        sa_layers.append(PointNetSequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers

def create_pointnet2_fp_modules(
    fp_blocks,
    in_channels,
    sa_in_channels,
    force_att=0,
    emb_dim=None,
    context_dim=None,
    use_att=False,
    dropout=0.1,
    with_se=False,
    normalize=True,
    eps=0,
    width_multiplier=1,
    voxel_resolution_multiplier=1
    ):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    c = 0

    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(
                in_channels=in_channels + sa_in_channels[-1 - fp_idx],
                out_channels=out_channels,
                emb_dim=emb_dim
            )
        )
        in_channels = out_channels[-1]

        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = ((c+1) % 2 == 0 and use_att and p == 0) or (force_att)
                if voxel_resolution is None:
                    block = functools.partial(SharedMLP, emb_dim=emb_dim)
                else:
                    block = functools.partial(
                        PVConv, 
                        kernel_size=3,
                        resolution=int(vr * voxel_resolution), 
                        attention=attention,
                        dropout=dropout,
                        with_se=with_se,
                        normalize=normalize,
                        eps=eps,
                        emb_dim=emb_dim,
                        context_dim=context_dim
                    )
                
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels

        fp_layers.append(PointNetSequential(*fp_blocks))

        c += 1
    
    return fp_layers, in_channels