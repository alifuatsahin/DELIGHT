from models import VAE, Prior
from .base_trainer import BaseTrainer
from utils.ema import EMA

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
            cfg
        )
        
        self.train_loader, self.test_loader = self.build_data()

        logger.info('done init trainer @{}', self.device)

    def resume(self, path, eval=False):
        ckpt = torch.load(path)
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
        if self.use_ema:
            data['ema'] = self.ema.state_dict()
        save_dir = self.cfg.save_dir if save_dir is None else save_dir
        save_name = "epoch_%s_iters_%s.pt" % (epoch, step) if save_name is None else save_name
        path = os.path.join(save_dir, "checkpoints", save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        torch.save(data, path)

        return path

    def build_model(self):
        cfg, args = self.cfg, self.args

        assert args.vae_checkpoint is not None, "VAE checkpoint must be provided for prior training"

        if args.distributed:
            dist.barrier()

        self.vae = VAE(cfg).to(self.device)  # Use eval mode for VAE during training
        ckpt = torch.load(args.vae_checkpoint, weights_only=False)
        vae_ckpt = self.filter_name(ckpt['model'])
        self.vae.load_state_dict(vae_ckpt)

        if 'ema' in ckpt.keys():
            ema_vae = EMA(self.vae)
            ema_ckpt = self.filter_name(ckpt['ema'])
            ema_vae.load_state_dict(ema_ckpt)
            ema_vae.copy_to(self.vae)

        self.vae.eval()  # Set VAE to eval mode
        utils.requires_grad(self.vae, False)  # Freeze VAE weights

        self.model = Prior(cfg).to(self.device)

        logger.info('Model initialized with num parameters: {}', sum(p.numel() for p in self.model.parameters()))
        
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)
        
        if self.use_ema:
            self.ema = EMA(self.model, decay=cfg.training.opt.ema_decay)
            logger.info('Using EMA with decay={}', cfg.training.opt.ema_decay)
    
    def train_iter(self, batch, step):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """
        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = batch['tr_points'].to(self.device)  # (B, Npoints, 3)
        latent, _, _ = self.vae.encode(tr_pts)

        with autocast(self.device_str, enabled=True):
            loss, logs_dict = self.model(latent)

            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        if self.use_ema and self.args.global_rank == 0:
            self.ema(self.model)

        # Log metrics efficiently
        if self.writer is not None and step is not None:
            for k, v in logs_dict.items():
                v0 = v.mean().detach().cpu().item() if torch.is_tensor(v) else v
                self.writer.avg_meter(k, v0, step=step)

        return lossv

    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        # Use EMA weights if available
        with self.ema_scope():
            was_training = self.model.training
            if was_training:
                self.model.eval()
            try:
                if hasattr(self.model, 'module'):
                    latents = self.model.module.sample(batch_size=n_samples)
                else:
                    latents = self.model.sample(batch_size=n_samples)

                # Decode the latents using the VAE
                samples = self.vae.decode(latents, n_sampled_points=n_sampled_points)
                # output = samples.permute(0, 2, 1).contiguous()  # B3N->BN3
            finally:
                if was_training:
                    self.model.train()

        return samples

    @torch.no_grad()
    def eval(self, x):
        """ 
        Evaluate the model on the given input x (reconstruction)
        
        Args:
            x: input point clouds
        """
        # For reconstruction evaluation, typically use current weights to measure training progress
        samples = self.vae.recont(x)
        # samples = samples.permute(0, 2, 1).contiguous() # B3N -> BN3

        return samples