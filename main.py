from default_config import cfg as config
import torch
import torch.nn as nn
import os
import numpy as np
from loguru import logger
import argparse
from utils.render_mitsuba_pc import pts2png
from models.vae import VAE

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
    """Remove 'module.' prefix from checkpoint keys."""
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

def normalize_pc(pc):
    """
    Normalize point cloud to zero mean and unit standard deviation.
    
    Args:
        pc: Point cloud tensor of shape (B, N, 3)
        
    Returns:
        Normalized point cloud
    """
    B, N = pc.shape[:2]
    pc_mean = pc.mean(axis=1).reshape(B, 1, -1)
    logger.info('Point cloud shape: {}. Computing mean over axis=1', pc.shape)
    pc_std = pc.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
    pc = (pc - pc_mean) / pc_std
    return pc

def rotate_pointcloud(pc, axis="z", angle_deg=90):
    """
    Rotate a point cloud around x, y, or z axis.
    
    Args:
        pc: (N, 3) or (B, N, 3) tensor
        axis: 'x', 'y', or 'z'
        angle_deg: rotation angle in degrees
        
    Returns:
        Rotated point cloud
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

    return pc @ R.T

def generate_trajectory(x0, x1, num_steps=100):
    """
    Generate a linear trajectory from x0 to x1.
    
    Args:
        x0: Starting point cloud (B, N, 3)
        x1: Target point cloud (B, N, 3)
        num_steps: Number of interpolation steps
        
    Returns:
        Trajectory tensor of shape (num_steps, B, N, 3)
    """
    velocity = x1 - x0
    trajectory = []
    
    for i in range(num_steps):
        alpha = (i + 1) / num_steps
        xt = x0 + alpha * velocity
        trajectory.append(xt.unsqueeze(0))
    
    return torch.cat(trajectory, dim=0)


def main(args):
    """Main function for point cloud generation and visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: {}', device)

    # Load configuration
    if not os.path.exists(args.config):
        logger.error('Config file not found: {}', args.config)
        return
    
    config.merge_from_file(args.config)

    # Load model if checkpoint is provided
    if args.checkpoint:
        logger.info('Loading model from checkpoint: {}', args.checkpoint)
        model = VAE(config).eval().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load input point cloud
    if not os.path.exists(args.input_pc):
        logger.error('Input point cloud file not found: {}', args.input_pc)
        return
        
    x1 = torch.from_numpy(np.load(args.input_pc)).float().unsqueeze(0)
    x1 = normalize_pc(x1)
    
    # Apply rotations if specified
    if args.rotate_z:
        x1 = rotate_pointcloud(x1.squeeze(), axis="z", angle_deg=args.rotate_z).unsqueeze(0)
    if args.rotate_x:
        x1 = rotate_pointcloud(x1.squeeze(), axis="x", angle_deg=args.rotate_x).unsqueeze(0)
    if args.rotate_y:
        x1 = rotate_pointcloud(x1.squeeze(), axis="y", angle_deg=args.rotate_y).unsqueeze(0)
    
    x1 = x1.to(device)

    # Generate initial random point cloud
    x0 = torch.randn(1, x1.shape[1], 3, device=device).float()

    # Generate trajectory
    logger.info('Generating trajectory with {} steps', args.num_steps)
    traj = generate_trajectory(x0, x1, num_steps=args.num_steps)

    # Save visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    filenames = [os.path.join(args.output_dir, f"frame_{i:03d}.png") for i in range(traj.shape[0])]

    logger.info('Rendering {} frames to {}', len(filenames), args.output_dir)
    pts2png(
        input_pts=traj.squeeze(), 
        file_name=filenames,
    )

    logger.info('Trajectory shape: {}', traj.shape)
    logger.info('Generation complete!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and visualize point cloud trajectories.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the config YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--input_pc', type=str, required=True,
                        help='Path to input point cloud (.npy file)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save output visualizations')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of trajectory steps')
    parser.add_argument('--rotate_x', type=float, default=0,
                        help='Rotation angle around X axis in degrees')
    parser.add_argument('--rotate_y', type=float, default=0,
                        help='Rotation angle around Y axis in degrees')
    parser.add_argument('--rotate_z', type=float, default=0,
                        help='Rotation angle around Z axis in degrees')
    
    args = parser.parse_args()
    main(args)
