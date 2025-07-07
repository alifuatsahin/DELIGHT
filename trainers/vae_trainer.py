import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from loguru import logger

from model.vae import VAE
from .base_trainer import BaseTrainer
from modules.losses import FlowMixtureLoss
from utils import utils

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.model = VAE(cfg, args)
        
        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize loss function
        self.loss_func = FlowMixtureLoss(
            pnll_weight=cfg.training.opt.pnll_weight,
            gnll_weight=cfg.training.opt.gnll_weight,
            gent_weight=cfg.training.opt.entl_weight,
            n_components=cfg.model.n_flows
        )

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

        model = VAE(cfg, args)

        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], output_device=args.rank)

        return model

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
            output_encoder, output_decoder, mixture_weights_logits = self.model(eval_pts, tr_pts)
            loss_dict = self.get_loss(output_encoder, output_decoder, mixture_weights_logits, writer=self.writer, it=step)

            loss = loss_dict['loss']
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(),
                                self.args.distributed)

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
            _, samples, labels, _ = self.model.sample(n_sampled_points, n_samples, deterministic=True)
            output = samples.permute(0, 2, 1).contiguous()  # BN3->B3N
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
        
        samples, labels, mixture_weights_logits = self.model.recont(x, deterministic=True)

        return samples, labels, mixture_weights_logits
    
    def get_loss(self, output_encoder, output_decoder, mixture_weights_logits, writer=None, it=None):
        loss, pnll, gnll, entl = self.loss_func(
            output_prior=output_encoder,
            output_decoder=output_decoder,
            mixture_weights_logits=mixture_weights_logits
        )

        output = {
            'loss': loss,
            'PNNL': pnll,
            'GNNL': gnll,
            'ENTL': entl
        }

        return output
    