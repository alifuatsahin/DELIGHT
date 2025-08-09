from models import VAE, DDPM
from .base_trainer import BaseTrainer

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

    def filter_name(self, ckpt):
        ckpt_new = {}
        for k, v in ckpt.items():
            if k[:7] == 'module.':
                kn = k[7:]
            elif k[:13] == 'model.module.':
                kn = k[13:]
            else:
                kn = k
            ckpt_new[kn] = v
        return ckpt_new

    def resume(self, path, eval=False):
        ckpt = torch.load(path, weights_only=True)
        ckpt = self.filter_name(ckpt)
        self.model.load_state_dict(ckpt['model'])
        self.vae.load_state_dict(ckpt['vae'])
        if not eval:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.grad_scalar.load_state_dict(ckpt['grad_scalar'])
            self.start_epoch = ckpt['epoch'] + 1
            self.step = ckpt['step']
        
        logger.info(f"Resumed from {path}")

    def save(self, epoch=None, step=None, save_dir=None, save_name=None):
        data = {
            'optimizer': self.optimizer.state_dict(),
            'vae': self.vae.state_dict(),
            'model': self.model.state_dict(),
            'grad_scalar': self.grad_scalar.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        save_dir = self.cfg.save_dir if save_dir is None else save_dir
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step) if save_name is None else save_name
        path = os.path.join(save_dir, "checkpoints", save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(data, path)

        return path

    def build_model(self):
        cfg, args = self.cfg, self.args

        assert args.vae_checkpoint is not None, "VAE checkpoint must be provided for DDPM training"

        if args.distributed:
            dist.barrier()

        self.vae = VAE(cfg).eval().to(self.device)  # Use eval mode for VAE during training
        self.vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=self.device)["model"], strict=True)

        self.model = DDPM(cfg).to(self.device)
        
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
        with torch.no_grad():
            latent, _, _ = self.vae.encode(tr_pts)

        with autocast(self.device, enabled=True):
            loss = self.model(latent)

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

        was_training = self.model.training
        if was_training:
            self.model.eval()

        try:
            # Sample latents using diffusion model
            latents = self.model.sample(batch_size=n_samples)
            
            # Decode the latents using the VAE
            samples, labels = self.vae.decode(latents, n_sampled_points=n_sampled_points)

        finally:
            if was_training:
                self.model.train()
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels

    @torch.no_grad()
    def eval(self, x):
        """ evaluate the model on the given input x """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        was_training = self.model.training
        if was_training:
            self.model.eval()
        
        try:
            samples, labels = self.vae.recont(x)
        finally:
            if was_training:
                self.model.train()
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels