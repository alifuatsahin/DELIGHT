from latent_diffusion.ldm.models.diffusion.ddpm import LatentDiffusion
from model.vae import VAE
from .base_trainer import BaseTrainer

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from loguru import logger
from utils import utils

class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

        self.model, self.vae = self.build_model()

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

        logger.info('done init trainer @{}', self.device)

    def build_model(self):
        cfg, args = self.cfg, self.args

        if args.distributed:
            dist.barrier()

        vae = VAE(cfg, args).eval().to(self.device)  # Use eval mode for VAE during training
        vae.load_state_dict(torch.load(cfg.vae_checkpoint, map_location=self.device)["model_state_dict"], strict=True)

        if cfg.model.ldm_backbone == "unet1":
            diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                                "params": {"dims": 1, "in_channels": 1, "model_channels": 256, "up_down_sampling": True,
                                            "attention_resolutions": (2, 4, 8), "channel_mult": (1, 2, 2, 4), "num_res_blocks": 2}}
        elif cfg.model.ldm_backbone == "unet1x":
            diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                                "params": {"dims": 1, "in_channels": 1, "model_channels": 320, "up_down_sampling": True,
                                            "attention_resolutions": (2, 4, 8), "channel_mult": (1, 2, 4, 4), "num_res_blocks": 3}}
        elif args.ldm_backbone == "unet1024":
            diff_model_config = {"target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
                                "params": {"dims": 1, "in_channels": 1024, "model_channels": 1024,
                                            "up_down_sampling": False}}
        else:
            raise NotImplementedError
        
        ddpm = LatentDiffusion(diff_model_config=diff_model_config, conditioning_key=None).to(self.device)

        if args.distributed:
            ddpm = nn.parallel.DistributedDataParallel(ddpm, device_ids=[args.rank], output_device=args.rank)

        return ddpm, vae
    
    def train_iter(self, batch, step):
        """ forward one iteration; and step optimizer  
        Args:
            data: (dict) tr_points shape: (B,N,3)
        """
        self.model.train()
        self.optimizer.zero_grad()

        tr_pts = batch['cloud'].to(self.device)  # (B, Npoints, 3)

        with autocast(self.device, enabled=True):
            output_encoder = self.vae.encode(tr_pts)
            loss = self.model(output_encoder['g_posterior_mus'], None)

            lossv = loss.detach().cpu().item()

        self.grad_scalar.scale(loss).backward()
        utils.average_gradients(self.model.parameters(),
                                self.args.distributed)

        self.grad_scalar.step(self.optimizer)
        self.grad_scalar.update()

        # Log metrics
        if self.writer is not None and step is not None:
            v0 = loss.mean().detach().cpu().item() if torch.is_tensor(loss) else loss
            self.writer.avg_meter("loss", v0, step=step)

        return lossv

    @torch.no_grad()
    def sample(self, n_sampled_points, n_samples=1):
        """ sample from the model """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
        try:
            # Generate seeds if not provided
            seeds = range(n_samples)
            
            # Create latent shape based on UNet backbone
            latent_dim = self.cfg.model.latent_dim  # 128
            latent_shape = (latent_dim,)  # Fallback
            
            # Generate initial noise
            x_T = torch.stack([torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed)) 
                              for seed in seeds], dim=0).to(self.device)
            
            # Sample latents using diffusion model
            latents = self.model.p_sample_loop(None, shape=(n_samples, *latent_shape), x_T=x_T, verbose=False)
            
            # Ensure latents are in correct shape for VAE decoder [B, latent_dim]
            if latents.dim() > 2:
                latents = latents.squeeze()  # Remove extra dimensions if needed
            
            # Decode the latents using the VAE
            with torch.no_grad():
                samples, labels, _ = self.vae.decoder.decode(latents, n_sampled_points=n_sampled_points)
                
        finally:
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels
    
    def ldm_sampling(model, batch_size, cond=None, verbose=False, seeds=None, latent_dim=1024):
        latent_shape = (latent_dim,)
        if seeds is None:
            seeds = range(batch_size)
        x_T = torch.stack([torch.randn(latent_shape, generator=torch.Generator().manual_seed(seed)) for seed in seeds], dim=0).to(model.device)
        return model.p_sample_loop(cond, shape=(batch_size, *latent_shape), x_T=x_T, verbose=verbose)

    @torch.no_grad()
    def eval(self, x):
        """ evaluate the model on the given input x """
        # Use EMA weights if available
        if self.cfg.training.opt.ema:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
        try:
            samples, labels, mixture_weights_logits = self.vae.recont(x, deterministic=True)
        finally:
            # Always restore original parameters
            if self.cfg.training.opt.ema:
                self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        return samples, labels, mixture_weights_logits