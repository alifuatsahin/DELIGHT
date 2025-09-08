from default_config import cfg as config
import torch
import os
from loguru import logger
import matplotlib.pyplot as plt
from modules.flows import PVCNN2Unet, timestep_embedding
from torchinfo import summary

def add_module_prefix(state_dict):
    """Add 'module.' prefix to every key in a state dict."""
    state_dict_new = {}
    for k, v in state_dict.items():
        # Avoid double prefixing if already present
        if not k.startswith('module.'):
            kn = 'module.' + k
        else:
            kn = k
        state_dict_new[kn] = v
    return state_dict_new

def filter_name(ckpt):
    ckpt_new = {}
    for k, v in ckpt.items():
        if k[:7] == 'module.':
            kn = k[7:]
        elif k[:13] == 'model.module.':
            kn = k[13:]
        elif k[:6] == 'module':
            kn = k[6:]
        else:
            kn = k
        ckpt_new[kn] = v
    return ckpt_new

def plot_pcs(pcs, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    for i, pc in enumerate(pcs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c='r')
        ax.set_title(f"Sample {i+1}")
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f"sample_{i+1}.png"))
        plt.close(fig)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SA_BLOCKS = [ # conv_configs, sa_configs
        ((32, 2, 32), (1024, 0.1, 32, (16, 32))),
        ((64, 1, 16), (256, 0.2, 32, (32, 64))),
        ((128, 1, 8), (64, 0.4, 32, (64, 128))),
    ]
    FP_BLOCKS = [
        ((128, 128), (128, 1, 8)), # fp_configs, conv_configs
        ((128, 128), (64, 1, 16)),
        ((128, 128, 64), (32, 2, 32)),
    ]

    model = PVCNN2Unet(emb_dim=6, context_dim=16, extra_feature_channels=0, sa_blocks=SA_BLOCKS, fp_blocks=FP_BLOCKS).to(device)

    logger.info('Model initialized with num parameters: {}', sum(p.numel() for p in model.parameters()))

    x = torch.randn(2, 3, 1024).to(device)  # Example input
    temb = torch.rand(2, 1).to(device)
    context = torch.rand(2, 16, 64).to(device)
    summary(model, input_data=(x, temb, context))

    x = model(x, temb=temb, context=context)

    print("Output shape:", x.shape)