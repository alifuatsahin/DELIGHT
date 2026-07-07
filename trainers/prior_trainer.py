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
        ckpt = torch.load(path, weights_only=False)
        if self.args.distributed:
            model_ckpt = self.add_module_prefix(ckpt['model'])
        else:
            model_ckpt = self.filter_name(ckpt['model'])
        self.model.load_state_dict(model_ckpt)
        self.vae.load_state_dict(self.filter_name(ckpt['vae']))
        self.load_ema_state(ckpt.get('ema'))
        self.step = ckpt.get('step', 0) or 0
        self.best_eval_score = ckpt.get('best_eval_score', -1)
        self.best_eval_epoch = ckpt.get('best_eval_epoch', 0)
        if not eval:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.grad_scalar.load_state_dict(ckpt['grad_scalar'])
            self.start_epoch = ckpt['epoch'] + 1

        logger.info(f"Resumed from {path} (step={self.step})")

    def save(self, epoch=None, step=None, save_dir=None, save_name=None):
        data = {
            'optimizer': self.optimizer.state_dict(),
            'vae': self.vae.state_dict(),
            'model': self.model.state_dict(),
            'grad_scalar': self.grad_scalar.state_dict(),
            'epoch': epoch,
            'step': step,
            'best_eval_score': self.best_eval_score,
            'best_eval_epoch': self.best_eval_epoch,
        }
        if self.use_ema and self.ema is not None:
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

        restoring = args.pretrained is not None or args.resume or args.eval
        assert args.vae_checkpoint or restoring, (
            "VAE checkpoint must be provided for prior training (pass "
            "--vae_checkpoint /path/to/vae.pt). It is optional only when "
            "resuming or evaluating a prior checkpoint, which already "
            "embeds the VAE weights.")

        if args.distributed:
            dist.barrier()

        self.vae = VAE(cfg).to(self.device)  # Use eval mode for VAE during training

        if args.vae_checkpoint:
            ckpt = torch.load(args.vae_checkpoint, weights_only=False)
            vae_ckpt = self.filter_name(ckpt['model'])
            self.vae.load_state_dict(vae_ckpt)

            if 'ema' in ckpt.keys():
                ema_vae = EMA(self.vae)
                ema_ckpt = self.filter_name(ckpt['ema'])
                ema_vae.load_state_dict(ema_ckpt)
                ema_vae.copy_to(self.vae)
            logger.info('Loaded frozen VAE from {}', args.vae_checkpoint)
        else:
            logger.info('No --vae_checkpoint given; the VAE weights will be '
                        'restored from the prior checkpoint at {}',
                        args.pretrained)

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

        if self.use_ema and self.ema is not None:
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
    def validate(self, step=None):
        """Held-out flow-matching loss for the prior. Lower is better.

        The noise and time draws inside Prior.forward are reseeded from a
        fixed seed so the number is comparable across epochs rather than
        dominated by Monte-Carlo variance.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        was_training = self.model.training
        self.model.eval()

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        seed = int(getattr(self.cfg, 'eval_seed', 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        total_loss, n_seen = 0.0, 0
        try:
            # Score the EMA weights, since those are what sampling uses.
            with self.ema_scope():
                for batch in self.test_loader:
                    x = batch['tr_points'].to(self.device)
                    latent, _, _ = self.vae.encode(x)
                    loss, _ = model(latent)
                    total_loss += loss.item() * x.shape[0]
                    n_seen += x.shape[0]
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            if was_training:
                self.model.train()

        score = total_loss / max(n_seen, 1)
        logger.info('[Prior val] flow-matching loss={:.6f} over {} shapes',
                    score, n_seen)
        if self.writer is not None:
            self.writer.add_scalar('eval/prior_fm_loss', score, step)

        return score

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