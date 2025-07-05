import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from loguru import logger

from model.vae import VAE
from .base_trainer import BaseTrainer
from modules.losses import Flow_Mixture_Loss
from utils import utils

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.model = VAE(cfg, args)

        self.loss_func = Flow_Mixture_Loss(
            pnll_weight=cfg.training.opt.pnll_weight,
            gnll_weight=cfg.training.opt.gnll_weight,
            gent_weight=cfg.training.opt.gent_weight,
            n_components=cfg.model.n_flows
        )

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
        see get_loss in models/shapelatent_diffusion.py 
        """
        self.model.train()

        warmup_iters = len(self.train_loader) * self.cfg.training.opt.vae_lr_warmup_epochs
        
        utils.update_vae_lr(self.cfg, step, warmup_iters, self.optimizer)

        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = data['tr_points'].to(self.device)  # (B, Npoints, 3)
        batch_size = tr_pts.size(0)

        with autocast(enabled=True):
            output_encoder, output_decoder, mixture_weights_logits = self.model(tr_pts, tr_pts)
            res = self.get_loss(output_encoder, output_decoder, mixture_weights_logits, writer=self.writer, it=step)

            loss = res['loss'].mean()
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(),
                                self.args.distributed)

        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        output = {}
        if self.writer is not None:
            for k, v in res.items():
                if 'print/' in k and step is not None:
                    v0 = v.mean().item() if torch.is_tensor(v) else v
                    self.writer.avg_meter(k.split('print/')[-1], v0,
                                          step=step)
                if 'hist/' in k:
                    output[k] = v

        output.update({
            'loss': lossv,
            'x_0_pred': res['x_0_pred'].detach().cpu(),  # perturbed data
            'x_0': res['x_0'].detach().cpu(),
            'x_t': res['final_pred'].detach().view(batch_size, -1, res['x_0'].shape[-1]),
            't': res.get('t', None)
        })
        for k, v in res.items():
            if 'vis/' in k or 'msg/' in k:
                output[k] = v

        return output
    
    def get_loss(self, output_encoder, output_decoder, mixture_weights_logits, writer=None, it=None):
        return self.loss_func(
            output_prior=output_encoder,
            output_decoder=output_decoder,
            mixture_weights_logits=mixture_weights_logits
        )
    