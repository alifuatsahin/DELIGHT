from default_config import cfg as config
import torch
import torch.nn as nn
import os
import numpy as np
from loguru import logger
import torchdiffeq
import matplotlib.pyplot as plt
from utils.render_mitsuba_pc import pts2png
from models.vae import VAE
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

def normalize_pc(pc):
    B, N = pc.shape[:2]
    pc_mean = pc.mean(axis=1).reshape(
        B, 1, -1)
    logger.info('all_points shape: {}. mean over axis=1',
                pc.shape)
    pc_std = pc.reshape(
        B, -1).std(axis=1).reshape(B, 1, 1)
    pc = (pc - pc_mean) / pc_std
    return pc

class Model(nn.Module):
    def __init__(self, x1):
        super(Model, self).__init__()
        self.x1 = x1
    
    def forward(self, x0, t):
        return self.x1 - x0
    
def rotate_pointcloud(pc, axis="z", angle_deg=90):
    """
    Rotate a point cloud around x, y, or z axis.
    pc: (N, 3) tensor
    axis: 'x', 'y', or 'z'
    angle_deg: rotation angle in degrees
    """
    angle = torch.tensor(angle_deg * torch.pi / 180.0)

    if axis == "x":
        R = torch.tensor([[1, 0, 0],
                          [0, torch.cos(angle), -torch.sin(angle)],
                          [0, torch.sin(angle), torch.cos(angle)]])
    elif axis == "y":
        R = torch.tensor([[torch.cos(angle), 0, torch.sin(angle)],
                          [0, 1, 0],
                          [-torch.sin(angle), 0, torch.cos(angle)]])
    elif axis == "z":
        R = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                          [torch.sin(angle), torch.cos(angle), 0],
                          [0, 0, 1]])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    return pc @ R.T   # (N, 3)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CHKPT_PATH = "../experiments/vae_softvq/pasta_bs64_20250913_023405/checkpoints/best_model.pth"
    cfg_path = os.path.dirname(CHKPT_PATH) + '/../config.yml'
    config.merge_from_file(cfg_path)

    # model = VAE(config).eval().to(device)

    x0 = torch.randn(1, 2048, 3, device=device).float()  # Initial point cloud
    x1 = torch.from_numpy(np.load("farfalle.npy")).float().unsqueeze(0)  # Target point cloud
    x1 = normalize_pc(x1)
    x1 = rotate_pointcloud(x1.squeeze(), axis="z", angle_deg=90)
    x1 = rotate_pointcloud(x1, axis="x", angle_deg=-90).to(device).unsqueeze(0)

    # model = Model(x1=x1).eval().to(device)

    # traj = model.recont(x1, return_trajectory=True)

    # traj = torchdiffeq.odeint(
    #     lambda t, x: model.forward(x, t.expand(x.shape[0])),
    #     x0,
    #     torch.linspace(0, 1, 100, device=x0.device),
    #     atol=1e-4,
    #     rtol=1e-4,
    #     method='dopri5',
    # )

    velocity = x1 - x0
    num_steps = 100

    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        xt = x0 + alpha * velocity
        if i == 0:
            traj = xt.unsqueeze(0)
        else:
            traj = torch.cat((traj, xt.unsqueeze(0)), dim=0)
    
    # plot_pcs([x0.squeeze().cpu().numpy(), final_pc.squeeze().cpu().numpy(), x1.squeeze().cpu().numpy()], out_dir="outputs")

    # file_names = ["outputs/final_reconstruction.png"]
    # pts2png(
    #     input_pts=final_pc, 
    #     file_name=file_names,
    # )

    filenames = ["outputs/frame_{:03d}.png".format(i) for i in range(traj.shape[0])]

    pts2png(
        input_pts=traj.squeeze(), 
        file_name=filenames,
    )

    print("Trajectory shape:", traj.shape)
