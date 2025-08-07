from models import VAE
from .base_trainer import BaseTrainer
from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion
from models.ddpm import DDPM

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from loguru import logger
from utils import utils
import os

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.build_model()

        # Initialize gradient scaler for mixed precision training
        self.grad_scalar = GradScaler(2**10, enabled=True)

        # Distributed training synchronization
        if args.distributed:
            logger.info('Waiting for barrier, device={}', self.device)
            dist.barrier()
            logger.info('Passed barrier, device={}', self.device)

        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = utils.get_opt(
            self.model.parameters(),
            cfg.training.opt,
            cfg.training.opt.ema, 
            cfg
        )
        
        self.train_loader, self.test_loader = self.build_data()

        logger.info('done init trainer @{}', self.device)

    def save(self, save_name=None, epoch=None, step=None, save_dir=None):
        grad_scalar = self.grad_scalar
        content = {'epoch': epoch, 'global_step': step, 
                   'grad_scalar': grad_scalar.state_dict(),
                   'ddpm_state_dict': self.model.state_dict(), 'ddpm_optimizer': self.optimizer.state_dict(),
                   'ddpm_scheduler': self.scheduler.state_dict(), 'vae_state_dict': self.vae.state_dict(),
                   }
        save_name = f"epoch_{epoch}_iters_{step}.pt" if save_name is None else save_name
        if save_dir is None:
            save_dir = self.cfg.save_dir
        path = os.path.join(save_dir, "checkpoints", save_name)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info('Save model to: {}', path)
        torch.save(content, path)
        
        return path

    def resume(self, path, eval=False):
        checkpoint = torch.load(path, map_location='cpu')
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['ddpm_state_dict'])

        # load dae
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['ddpm_optimizer'])
        self.scheduler.load_state_dict(checkpoint['ddpm_scheduler'])

        # load vae
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae = self.vae.to(self.device)

        # need to comment if load regular vae from voxel2input_ada trainer
        self.grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        self.step = checkpoint['global_step']

        logger.info('Resumed from : {}, epoch={}', path, self.start_epoch)

    def build_model(self):
        cfg, args = self.cfg, self.args

        if args.distributed:
            dist.barrier()

        self.vae = VAE(cfg).eval().to(self.device)  # Use eval mode for VAE during training
        self.vae.load_state_dict(torch.load(cfg.vae_checkpoint, map_location=self.device)["model"], strict=True)

        self.model = DDPM(cfg.ddpm, quantizer=cfg.vae.quantizer, ch=cfg.vae.soft_vq.e_dim).to(self.device)
        
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    def train_iter(self, batch, step):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """
        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = batch['cloud'].to(self.device)  # (B, Npoints, 3)

        with autocast(self.device, enabled=True):
            latent, _, _ = self.vae.encode(tr_pts)
            loss = self.model(latent, None)

            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        # Log metrics
        if self.writer is not None and step is not None:
            v0 = loss.mean().detach().cpu().item() if torch.is_tensor(loss) else loss
            self.writer.avg_meter("loss", v0, step=step)

        return lossv

    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
        try:
            # Generate seeds if not provided
            seeds = range(n_samples)
            
            # Create latent shape based on UNet backbone
            latent_dim = self.cfg.vae.latent_dim  # 128
            latent_shape = (latent_dim,)  # Fallback
            
            # Generate initial noise
            x_T = torch.stack([torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed)) 
                              for seed in seeds], dim=0).to(self.device)
            
            # Sample latents using diffusion model
            latents = self.model.p_sample_loop(None, shape=(n_samples, *latent_shape), x_T=x_T, verbose=False)
            
            # Ensure latents are in correct shape for VAE decoder [B, latent_dim]
            if latents.dim() > 2:
                latents = latents.squeeze()  # Remove extra dimensions if needed
            
            # Decode the latents using the VAE
            with torch.no_grad():
                samples, labels = self.vae.decoder.decode(latents, n_sampled_points=n_sampled_points)
                
        finally:
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels
    
    def ldm_sampling(model, batch_size, cond=None, verbose=False, seeds=None, latent_dim=1024):
        latent_shape = (latent_dim,)
        if seeds is None:
            seeds = range(batch_size)
        x_T = torch.stack([torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed)) for seed in seeds], dim=0).to(model.device)
        return model.p_sample_loop(cond, shape=(batch_size, *latent_shape), x_T=x_T, verbose=verbose)

    @torch.no_grad()
    def eval(self, x):
        """ evaluate the model on the given input x """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
        try:
            samples, labels = self.vae.recont(x)
        finally:
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels