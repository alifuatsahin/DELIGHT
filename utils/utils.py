import torch
import torch.distributed as dist
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import numpy as np
import types
import os

class Writer:
    def __init__(self, rank, save=None):
        self.rank = rank
        self.meter_dict = {}  # Initialize meter dictionary

        if rank == 0:  # Only initialize TensorBoard writer on rank 0
            logger.info(f"Init TensorBoard writer on rank {self.rank}")
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)
        else:
            self.writer = None

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            if 'step' in kwargs:
                self.writer.add_scalar(*args, global_step=kwargs['step'])
            else:
                self.writer.add_scalar(*args, **kwargs)

    def log_other(self, key, value):
        print(f"{key}: {value}")

    def add_figure(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_image(*args, **kwargs)
            # Remove flush() - TensorBoard handles this automatically

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.add_histogram(*args, **kwargs)

    def avg_meter(self, name, value, step=None):
        if self.rank == 0:
            if name not in self.meter_dict:
                self.meter_dict[name] = AverageMeter()
            self.meter_dict[name].update(value)

    def upload_meter(self, step=None):
        if self.rank == 0:  # Add rank check for efficiency
            for name, value in self.meter_dict.items():
                self.add_scalar(name, value.avg, step=step)
            self.meter_dict = {}

    def close(self, *args, **kwargs):
        if self.rank == 0 and self.writer is not None:
            self.writer.close()


def init(rank, seed=0, save_dir=None):
    logger.info('[INIT] at rank={}, seed={}', rank, seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True

    writer = Writer(rank, save_dir)
    logger.info('[INIT] DONE')

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

def get_opt(params, cfgopt, use_ema, other_cfg=None):
    # Create optimizer
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
    else:
        raise ValueError(f"Unsupported optimizer type: {cfgopt.type}")
        
    # Apply EMA if requested
    if use_ema:
        from .ema import EMA
        optimizer = EMA(optimizer, ema_decay=cfgopt.ema_decay)

    # Default scheduler (constant lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
    
    # Get scheduler type and create if specified
    scheduler_type = getattr(cfgopt, "scheduler", "")
    if not scheduler_type:  # Early return for empty scheduler
        return optimizer, scheduler
        
    logger.info('Creating scheduler_type: {}', scheduler_type)
    
    if scheduler_type == 'exponential':
        decay = float(getattr(cfgopt, "step_decay", 0.1))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
        
    elif scheduler_type == 'step':
        step_size = int(getattr(cfgopt, "step_epoch", 50))
        decay = float(getattr(cfgopt, "step_decay", 0.6))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay)
        
    elif scheduler_type == 'linear':
        start_epoch = int(getattr(cfgopt, 'sched_start_epoch', 200*1e3))
        end_epoch = int(getattr(cfgopt, 'sched_end_epoch', 400*1e3))
        end_lr = float(getattr(cfgopt, 'end_lr', 1e-4))
        start_lr = float(cfgopt.lr)

        def lambda_rule(epoch):
            if epoch <= start_epoch:
                return 1.0
            elif epoch <= end_epoch:
                total = end_epoch - start_epoch
                delta = epoch - start_epoch
                frac = delta / total
                return (1 - frac) + frac * (end_lr / start_lr)
            else:
                return end_lr / start_lr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif scheduler_type == 'lambda':
        step_size = int(getattr(cfgopt, "step_epoch", 2000))
        final_ratio = float(getattr(cfgopt, "final_ratio", 0.01))
        start_ratio = float(getattr(cfgopt, "start_ratio", 0.5))
        duration_ratio = float(getattr(cfgopt, "duration_ratio", 0.45))

        def lambda_rule(ep):
            return 1.0 - min(1, max(0, ep - start_ratio * step_size) / 
                           (duration_ratio * step_size)) * (1 - final_ratio)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif scheduler_type == 'cosine_anneal_nocycle':
        assert other_cfg is not None, "other_cfg required for cosine_anneal_nocycle scheduler"
        final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
        start_ratio = float(getattr(cfgopt, "start_ratio", 0.6))
        total_epoch = int(other_cfg.training.epochs)  # CRITICAL: Need total training duration
        
        eta_min = float(cfgopt.lr) * final_lr_ratio  # Minimum LR (1% of initial)
        eta_max = float(cfgopt.lr)                   # Maximum LR (initial LR)
        T_max = total_epoch * (1 - start_ratio)      # Cosine cycle duration

        def lambda_rule(ep):
            curr_ep = max(0., ep - start_ratio * total_epoch)
            # Apply cosine decay: lr = η_min + 0.5*(η_max - η_min)*(1 + cos(π*t/T_max))
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * curr_ep / T_max))
            return lr / eta_max
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif scheduler_type == 'cosine_anneal':
        assert other_cfg is not None, "other_cfg required for cosine_anneal scheduler"
        final_lr_ratio = float(getattr(cfgopt, "final_lr_ratio", 0.01))
        num_cycles = int(getattr(cfgopt, "num_cycles", 10))
        total_epoch = int(other_cfg.training.epochs)  # CRITICAL: Need total training duration
        T_max = total_epoch / num_cycles
        eta_min = float(cfgopt.lr) * final_lr_ratio
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return optimizer, scheduler

def init_processes(global_rank, size, args):
    # Set device and initialize process group
    if args.num_gpus >= 1:
        torch.cuda.set_device(global_rank)  # Use rank directly for spawn
        backend = 'nccl'
    else:
        backend = 'gloo'  # Fallback for CPU-only training
        
    if args.num_gpus > 1:
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '6020')
        logger.info('Set MASTER_ADDR: {}, MASTER_PORT: {}', os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        
        try:
            device_id = torch.device(f"cuda:{args.local_rank}")
            dist.init_process_group(
                backend=backend, init_method='env://', rank=global_rank, world_size=size, device_id=device_id)
            logger.info('Init Process: rank={}, world_size={}', global_rank, size)
        except Exception as e:
            logger.error('Failed to initialize process group: {}', e)
            raise
    else:
        logger.info('Single GPU training, no distributed initialization needed')
        
    logger.info('Process initialization completed for rank {}', global_rank)