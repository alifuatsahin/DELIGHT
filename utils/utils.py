import torch
import torch.distributed as dist
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np
import types

class Writer:
    def __init__(self, rank, save=None):
        self.rank = rank

        logger.info(f"Init TensorBoard writer on rank {self.rank}")
        self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, step=None):
        if self.rank == 0:
            self.writer.add_scalar(*args, global_step=step)

    def log_other(self, key, value):
        print(f"{key}: {value}")

    def add_figure(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_image(*args, **kwargs)
            self.writer.flush()

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_histogram(*args, **kwargs)

def init(rank, seed=0, save_dir=None):
    logger.info('[common-init] at rank={}, seed={}', rank, seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    writer = Writer(rank, save_dir)
    logger.info('INIT DONE')

    return writer

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        if isinstance(params, types.GeneratorType):
            params = [p for p in params]

        size = float(dist.get_world_size())
        grad_data = []
        grad_size = []
        grad_shapes = []
        # Gather all grad values
        for param in params:
            if param.requires_grad:
                if param.grad is not None:
                    grad_size.append(param.grad.data.numel())
                    grad_shapes.append(list(param.grad.data.shape))
                    grad_data.append(param.grad.data.flatten())
        grad_data = torch.cat(grad_data).contiguous()

        # All-reduce grad values
        grad_data /= size
        dist.all_reduce(grad_data, op=dist.ReduceOp.SUM)

        # Put back the reduce grad values to parameters
        base = 0
        i = 0
        for param in params:
            if param.requires_grad and param.grad is not None:
                param.grad.data = grad_data[base:base +
                                            grad_size[i]].view(grad_shapes[i])
                base += grad_size[i]
                i += 1


def get_opt(params, cfgopt, use_ema, other_cfg=None):
    if cfgopt.type == 'adam':
        optimizer = optim.Adam(params,
                               lr=float(cfgopt.lr),
                               betas=(cfgopt.beta1, cfgopt.beta2),
                               weight_decay=cfgopt.weight_decay)
    elif cfgopt.type == 'adamw':
        optimizer = optim.AdamW(params,
                                lr=float(cfgopt.lr),
                                betas=(cfgopt.beta1, cfgopt.beta2),
                                weight_decay=cfgopt.weight_decay)
        
    if use_ema:
        from .ema import EMA
        optimizer = EMA(optimizer, ema_decay=cfgopt.ema_decay)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: 1.0)  # constant lr
    
    scheduler_type = getattr(cfgopt, "scheduler", None)

    if scheduler_type is not None and len(scheduler_type) > 0:
        logger.info('get scheduler_type: {}', scheduler_type)
        if scheduler_type == 'exponential':
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        elif scheduler_type == 'step':
            step_size = int(getattr(cfgopt, "step_epoch", 500))
            decay = float(getattr(cfgopt, "step_decay", 0.1))
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=step_size,
                                                  gamma=decay)
        elif scheduler_type == 'linear':  # use default setting from shapeLatent
            start_epoch = int(getattr(cfgopt, 'sched_start_epoch', 200*1e3))
            end_epoch = int(getattr(cfgopt, 'sched_end_epoch', 400*1e3))
            end_lr = float(getattr(cfgopt, 'end_lr', 1e-4))
            start_lr = cfgopt.lr

            def lambda_rule(epoch):
                if epoch <= start_epoch:
                    return 1.0
                elif epoch <= end_epoch:
                    total = end_epoch - start_epoch
                    delta = epoch - start_epoch
                    frac = delta / total
                    return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
                else:
                    return end_lr / start_lr
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        elif scheduler_type == 'lambda':  # linear':
            step_size = int(getattr(cfgopt, "step_epoch", 2000))
            final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
            duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

            def lambda_rule(ep):
                lr_l = 1.0 - min(
                    1,
                    max(0, ep - start_ratio * step_size) /
                    float(duration_ratio * step_size)) * (1 - final_ratio)
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        elif scheduler_type == 'cosine_anneal_nocycle':
            ## logger.info('scheduler_type: {}', scheduler_type)
            assert(other_cfg is not None)
            final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
            eta_min = float(cfgopt.lr) * final_lr_ratio
            eta_max = float(cfgopt.lr)

            total_epoch = int(other_cfg.trainer.epochs)
            ##getattr(cfgopt, "step_epoch", 2000)
            start_ratio = float(getattr(cfgopt, "start_ratio", 0.6))
            T_max = total_epoch * (1 - start_ratio)

            def lambda_rule(ep):
                curr_ep = max(0., ep - start_ratio * total_epoch)
                lr = eta_min + 0.5 * (eta_max - eta_min) * (
                    1 + np.cos(np.pi * curr_ep / T_max))
                lr_l = lr / eta_max
                return lr_l

            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lr_lambda=lambda_rule)

        else:
            assert 0, "args.schedulers should be either 'exponential' or 'linear' or 'step'"

    return optimizer, scheduler