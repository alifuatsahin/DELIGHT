import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from loguru import logger

from model.vae import VAE
from .base_trainer import BaseTrainer
from modules.losses import FlowMixtureLoss
from utils import utils
from utils.utils import AverageMeter

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.model = VAE(cfg, args)

        self.loss_func = FlowMixtureLoss(
            pnll_weight=cfg.training.opt.pnll_weight,
            gnll_weight=cfg.training.opt.gnll_weight,
            gent_weight=cfg.training.opt.entl_weight,
            n_components=cfg.model.n_flows
        )

        self.LB = AverageMeter()
        self.PNLL = AverageMeter()
        self.GNLL = AverageMeter()
        self.GENT = AverageMeter()

        self.grad_scalar = GradScaler(2**10, enabled=True)

        if args.distributed:
            logger.info('waitting for barrier, device={}', self.device)
            dist.barrier()
            logger.info('pass barrier, device={}', self.device)

        self.optimizer, self.scheduler = utils.get_opt(
            self.model.parameters(),
            self.cfg.training.opt,
            cfg.training.opt.ema, self.cfg)
        
        self.train_loader, self.test_loader = self.build_data()

        logger.info('done init trainer @{}', self.device)

    def train_iter(self, data, step):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """
        self.model.train()

        warmup_iters = len(self.train_loader) * self.cfg.training.opt.vae_lr_warmup_epochs
        
        utils.update_vae_lr(self.cfg, step, warmup_iters, self.optimizer)

        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = data['tr_points'].to(self.device)  # (B, Npoints, 3)
        eval_pts = data['eval_pts'].to(self.device)  # (B, Npoints, 3)
        batch_size = tr_pts.size(0)

        with autocast(enabled=True):
            output_encoder, output_decoder, mixture_weights_logits = self.model(eval_pts, tr_pts)
            output = self.get_loss(output_encoder, output_decoder, mixture_weights_logits, writer=self.writer, it=step)

            loss = output['loss']
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(),
                                self.args.distributed)

        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        output = {}
        if self.writer is not None:
            for k, v in output.items():
                if 'print/' in k and step is not None:
                    v0 = v.mean().item() if torch.is_tensor(v) else v
                    self.writer.avg_meter(k.split('print/')[-1], v0)
                if 'hist/' in k:
                    output[k] = v

        output.update({
            'loss': lossv,
            'PNNL': output_decoder['PNNL'].detach().cpu(),  # perturbed data
            'GNNL': output_decoder['GNNL'].detach().cpu(),
            'ENTL': output_decoder['ENTL'].detach().cpu(),
        })

        return output
    
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        self.model.eval()
        with torch.no_grad():
            output = self.model.sample(n_sampled_points, n_samples)
        return output

    def eval(self, x):
        """ evaluate the model on the given input x """
        with torch.no_grad():
            samples, labels, mixture_weights_logits = self.model.recont(x)

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
    