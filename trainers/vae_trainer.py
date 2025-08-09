import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from loguru import logger
import os

from models.vae import VAE
from .base_trainer import BaseTrainer
from utils import utils

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

        logger.info('Done init trainer @{}', self.device)

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
        if not eval:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.grad_scalar.load_state_dict(ckpt['grad_scalar'])
            self.start_epoch = ckpt['epoch'] + 1
            self.step = ckpt['step']
        
        logger.info(f"Resumed from {path}")

    def save(self, epoch=None, step=None, save_dir=None, save_name=None):
        data = {
            'optimizer': self.optimizer.state_dict(),
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
        if args.distributed:
            dist.barrier()

        self.model = VAE(cfg).to(self.device)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


    def train_iter(self, batch, step):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """

        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = batch['tr_points'].to(self.device)  # (B, Npoints, 3)
        eval_pts = batch['te_points'].to(self.device)  # (B, Npoints, 3) - fallback to tr_pts if missing

        with autocast(self.device_str, enabled=True):
            logs_dict = self.model(eval_pts, tr_pts, step=step)

            loss = logs_dict['loss']
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

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
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        was_training = self.model.training
        self.model.eval()
        try:
            if hasattr(self.model, 'module'):
                samples, labels = self.model.module.sample(n_sampled_points, n_samples)
            else:
                samples, labels = self.model.sample(n_sampled_points, n_samples)
            output = samples.permute(0, 2, 1).contiguous()  # B3N->BN3
        finally:
            if was_training:
                self.model.train()
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
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        was_training = self.model.training
        self.model.eval()
        try:
            if hasattr(self.model, 'module'):
                samples, labels = self.model.module.recont(x)
            else:
                # For single GPU models
                samples, labels = self.model.recont(x)
            samples = samples.permute(0, 2, 1).contiguous() # B3N -> BN3
        finally:
            if was_training:
                self.model.train()
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels
    