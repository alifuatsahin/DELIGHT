import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from loguru import logger

from models.vae import VAE
from .base_trainer import BaseTrainer
from utils import utils

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.build_model(cfg, args)

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

        logger.info('Done init trainer @{}', self.device)

    def build_model(self):
        cfg, args = self.cfg, self.args
        if args.distributed:
            dist.barrier()

        self.model = VAE(cfg, args).to(self.device)  # Use eval mode for VAE during training

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
        eval_pts = batch.get('eval_cloud', tr_pts).to(self.device)  # (B, Npoints, 3) - fallback to tr_pts if missing

        with autocast(self.device, enabled=True):
            loss_dict = self.model(eval_pts, tr_pts)

            loss = loss_dict['loss']
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        # Log metrics efficiently
        if self.writer is not None and step is not None:
            for k, v in loss_dict.items():
                v0 = v.mean().detach().cpu().item() if torch.is_tensor(v) else v
                self.writer.avg_meter(k, v0, step=step)

        return lossv
    
    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
        try:
            samples, labels = self.model.sample(n_sampled_points, n_samples, deterministic=True)
            output = samples.permute(0, 2, 1).contiguous()  # B3N->BN3
        finally:
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return output, labels

    @torch.no_grad()
    def eval(self, x):
        """ 
        Evaluate the model on the given input x (reconstruction)
        
        Args:
            x: input point clouds
            use_ema: whether to use EMA weights
        """
        # For reconstruction evaluation, typically use current weights to measure training progress
        
        samples, labels = self.model.recont(x, deterministic=True)
        samples = samples.permute(0, 2, 1).contiguous() # B3N -> BN3

        return samples, labels
    