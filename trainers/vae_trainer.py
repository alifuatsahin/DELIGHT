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
            cfg
        )
        
        self.train_loader, self.test_loader = self.build_data()

        logger.info('Done init trainer @{}', self.device)

    def resume(self, path, eval=False):
        ckpt = torch.load(path)
        if self.args.distributed:
            model_ckpt = self.add_module_prefix(ckpt['model'])
        else:
            model_ckpt = self.filter_name(ckpt['model'])
        self.model.load_state_dict(model_ckpt)
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
        if args.distributed:
            dist.barrier()

        self.model = VAE(cfg).to(self.device)

        logger.info('Model initialized with num parameters: {}', sum(p.numel() for p in self.model.parameters()))
        logger.info('Encoder num parameters: {}', sum(p.numel() for p in self.model.encoder.parameters()))
        logger.info('Decoder num parameters: {}', sum(p.numel() for p in self.model.decoder.parameters()))

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        if self.use_ema and self.args.global_rank == 0:
            from utils.ema import EMA
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
        # eval_pts = batch['te_points'].to(self.device)  # (B, Npoints, 3) - fallback to tr_pts if missing

        with autocast(self.device_str, enabled=True):
            logs_dict = self.model(tr_pts, step=step)

            loss = logs_dict['loss']
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

        return output, labels

    @torch.no_grad()
    def eval(self, x):
        """ 
        Evaluate the model on the given input x (reconstruction)
        
        Args:
            x: input point clouds
        """
        with self.ema_scope():
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

        return samples, labels
    