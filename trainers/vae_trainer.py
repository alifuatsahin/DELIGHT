import torch
from torch.cuda.amp import autocast, GradScaler
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

        self.loss_func = FlowMixtureLoss(
            pnll_weight=cfg.training.opt.pnll_weight,
            gnll_weight=cfg.training.opt.gnll_weight,
            gent_weight=cfg.training.opt.entl_weight,
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
        """
        self.model.train()

        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = data['tr_points'].to(self.device)  # (B, Npoints, 3)
        eval_pts = data['eval_pts'].to(self.device)  # (B, Npoints, 3)

        with autocast(enabled=True):
            output_encoder, output_decoder, mixture_weights_logits = self.model(eval_pts, tr_pts)
            loss_dict = self.get_loss(output_encoder, output_decoder, mixture_weights_logits, writer=self.writer, it=step)

            loss = loss_dict['loss']
            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(),
                                self.args.distributed)

        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        output = {}
        if self.writer is not None:
            for k, v in loss_dict.items():
                if step is not None:
                    v0 = v.mean().item() if torch.is_tensor(v) else v
                    self.writer.avg_meter(k, v0)


        output.update({
            'loss': lossv,
            'PNNL': loss_dict['PNNL'].detach().cpu(),
            'GNNL': loss_dict['GNNL'].detach().cpu(),
            'ENTL': loss_dict['ENTL'].detach().cpu(),
        })

        return output
    
    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        if self.cfg.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        self.model.eval()

        _, samples, labels, _ = self.model.sample(n_sampled_points, n_samples)

        output = samples.permute(0, 2, 1).contiguous()  # BN3->B3N

        # switch back to original parameters
        if self.cfg.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return output, labels

    @torch.no_grad()
    def eval(self, x):
        """ evaluate the model on the given input x """
        if self.cfg.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        self.model.eval()

        samples, labels, mixture_weights_logits = self.model.recont(x)

        if self.cfg.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

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
    