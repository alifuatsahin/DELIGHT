# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch.nn as nn 
from modules.pvcnn2 import create_pointnet2_sa_components 
# implement the global encoder for VAE model 

class Encoder(nn.Module):
    # sa_blocks = [
    #     [[32, 2, 32], [1024, 0.1, 32, [32, 32]]],
    #     [[32, 1, 16], [256, 0.2, 32, [32, 64]]]
    #     ]
    sa_blocks = [
        [[32, 2, 32], [1024, 0.1, 32, [32, 64]]],
        [[64, 1, 16], [256, 0.2, 32, [64, 128]]],
        [[128, 1, 16], [128, 0.4, 64, [128, 512, 1024]]],
    ]
    force_att = 0 # add attention to all layers  
    def __init__(self, input_dim, extra_feature_channels=0):
        super().__init__()
        sa_blocks = self.sa_blocks 
        layers, sa_in_channels, channels_sa_features, _  = \
            create_pointnet2_sa_components(sa_blocks, 
            extra_feature_channels, input_dim=input_dim, 
            embed_dim=0, force_att=self.force_att,
            use_att=True, with_se=True)
        # channels_sa_features: the output feature dimension of the last SA layer

        self.out_features = channels_sa_features
        self.layers = nn.ModuleList(layers) 
        self.voxel_dim = [n[1][-1][-1] for n in self.sa_blocks]

    def forward(self, x):
        """
        Args: 
            x: B,N,3 
        Returns: 
            mu, sigma: B,D
        """
        output = {} 
        # x = x.transpose(1, 2) # B,3,N
        xyz = x ## x[:,:3,:]
        features = x
        for layer in self.layers:
            features, xyz, _ = layer( (features, xyz, None) )
        # features: B,D,N; xyz: B,3,N
       
        features = features.max(-1)[0]
        return features


