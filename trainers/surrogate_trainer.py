import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from contextlib import contextmanager
from loguru import logger
import numpy as np
import time
import os

from models.surrogate import Surrogate
from utils import utils
from utils.utils import AverageMeter

class Trainer:
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.build_model()
        self.writer = None
        self.use_ema = cfg.training.opt.ema

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

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model)
            if context is not None:
                logger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.ema.restore(self.model.parameters())
                if context is not None:
                    logger.info(f"{context}: Restored training weights")

    def log_loss(self, loss_dict, writer=None, step=None):
        """Log loss values to writer"""
        if writer is not None:
            for key, value in loss_dict.items():
                writer.add_scalar(f'train/{key}', value, step)

    def filter_name(self, ckpt):
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

    def add_module_prefix(self, state_dict):
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

    def set_writer(self, writer):
        self.writer = writer

    def epoch_end(self, epoch, writer=None):
        # Signal now that the epoch ends....
        if self.scheduler is not None:
            self.scheduler.step()
            if writer is not None:
                writer.add_scalar(
                    'train/opt_lr', self.scheduler.get_last_lr()[0], epoch)
        if writer is not None:
            writer.upload_meter(step=epoch)

    def build_data(self):
        logger.info('Building data loader...')
        if self.cfg.training.type == 'vae' and (self.cfg.vae.flow.cfm_method == 'ot' or self.cfg.vae.flow.cfm_method == 'schrodinger_bridge'):
            return_superset = True
        else:
            return_superset = False
        loaders = get_data_loaders(self.cfg.data, self.args, return_superset=return_superset)
        train_loader = loaders['train_loader']
        test_loader = loaders['test_loader']

        return train_loader, test_loader

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

        self.model = Surrogate(cfg).to(self.device)

        logger.info('Model initialized with num parameters: {}', sum(p.numel() for p in self.model.parameters()))
        logger.info('Encoder num parameters: {}', sum(p.numel() for p in self.model.encoder.parameters()))
        logger.info('Decoder num parameters: {}', sum(p.numel() for p in self.model.decoder.parameters()))

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank], output_device=args.local_rank)

        self.loss_fn = torch.nn.MSELoss() if cfg.surrogate.mse_loss == 'MSE' else torch.nn.L1Loss()

        if self.use_ema and self.args.global_rank == 0:
            from utils.ema import EMA
            self.ema = EMA(self.model, decay=cfg.training.opt.ema_decay)
            logger.info('Using EMA with decay={}', cfg.training.opt.ema_decay)

    def train_epochs(self):
        cfg, args = self.cfg, self.args
        writer = self.writer
        train_loader = self.train_loader

        logger.info('[GPU {}] Starting training for {} epochs'.format(args.local_rank, cfg.training.epochs))

        tic_global = time.time()
        if args.global_rank == 0:
            tic_log = time.time()
            start_time = time.time()
        avg_time = AverageMeter()
        step = 0

        self.total_iter = cfg.training.epochs * len(train_loader)
        if hasattr(self.model, 'module'):
            self.model.module.total_iter = self.total_iter
        else:
            # For single GPU models
            self.model.total_iter = self.total_iter

        for epoch in range(self.start_epoch, cfg.training.epochs + 1):
            self.model.train()

            if args.global_rank == 0:
                tic_epo = time.time()
            epoch_loss = []
           
            for idx, batch in enumerate(train_loader):
                step = idx + len(train_loader) * epoch

                if args.global_rank == 0 and self.writer is not None:
                    tic_iter = time.time()

                loss = self.train_iter(batch)

                if args.global_rank == 0:
                    epoch_loss.append(loss)

                if self.args.global_rank == 0 and (
                        time.time() - tic_log > 60
                ):  # log per min
                    logger.info(
                        f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.4f} | '
                        f'[exp] {cfg.save_dir} | [step] {step:5d}'
                    )
                    tic_log = time.time()

                # -- visualize rec and samples -- #
                if step % int(cfg.vis.log_freq) == 0 and args.global_rank == 0 and not step == 0:
                    avg_loss = np.array(epoch_loss).mean()
                    epoch_loss = []  # clean up epoch loss
                    self.log_loss({'epo_loss': avg_loss},
                                  writer=writer, step=step)

                # -- timer -- #
                if args.global_rank == 0 and self.writer is not None:
                    time_iter = time.time() - tic_iter
                    self.writer.avg_meter('time_iter', time_iter, step=step)

            if args.global_rank == 0 and self.writer is not None:
                epo_time = (time.time() - tic_epo) / 60.0  # min
                avg_time.update(epo_time)
                logger.info(
                    f'E{epoch} iter[{idx}/{len(train_loader)}] | [Loss] {np.array(epoch_loss).mean():.4f} | '
                    f'[exp] {cfg.save_dir} | [step] {step:5d} | [time] {epo_time:.1f}m (~{int(avg_time.avg * (cfg.training.epochs - epoch) / 60)}h) | '
                    f'[best] {self.best_eval_epoch} {self.best_eval_score * 1e2:.3f}x1e-2'
                )
                tic_log = time.time()

            if epoch % int(cfg.vis.save_freq) == 0 and int(cfg.vis.save_freq) > 0 and args.global_rank == 0:
                save_path = self.save(epoch=epoch, step=step)
                logger.info(f"Checkpoint saved at {save_path} [Epoch] {epoch}")
            
            if (time.time() - tic_global) / 60 > cfg.vis.save_time and args.global_rank == 0:
                save_path = self.save(epoch=epoch, step=step, save_name='snapshot.pth')
                logger.info(f"Checkpoint saved at {save_path}, [Time] {(time.time() - start_time) / 60}h")
                tic_global = time.time()

            if int(cfg.vis.val_freq) > 0 and epoch % int(cfg.vis.val_freq) == 0 and args.global_rank == 0:
                score = self.eval(step=epoch)
                if score < self.best_eval_score or self.best_eval_score < 0:
                    self.save(save_name='best_eval.pth',
                              epoch=epoch, step=step)
                    self.best_eval_score = score
                    self.best_eval_epoch = epoch

            self.epoch_end(epoch, writer=writer)

        if args.global_rank == 0:
            logger.info(f'Training finished, total time: {(time.time() - start_time) / 60:.2f} min')
            logger.info(f'Best eval score: {self.best_eval_score * 1e2:.3f}x1e-2 at epoch {self.best_eval_epoch}')
            self.writer.close() if hasattr(self, 'writer') else None


    def train_iter(self, batch):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """

        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = batch['tr_points'].to(self.device)  # (B, Npoints, 3)
        gt = batch['score'].to(self.device)  # (B, 1)

        with autocast(self.device_str, enabled=True):
            output = self.model(tr_pts)

            loss = self.loss_fn(output, gt)

            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        if self.use_ema and self.args.global_rank == 0:
            self.ema(self.model)

        return lossv

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
                    output = self.model.module(x)
                else:
                    # For single GPU models
                    output = self.model(x)
                # samples = samples.permute(0, 2, 1).contiguous() # B3N -> BN3
            finally:
                if was_training:
                    self.model.train()

        return output
    