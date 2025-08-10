# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
""" src: ddim/model/ema.py 
implement the EMA model 
usage: 
    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
    ema_helper.register(model)
    ema_helper.load_state_dict(states[-1])
    ema_helper.ema(model)

after optimizer.step()
    ema_helper.update(model)

copied and modified from 
    https://github.com/NVlabs/LSGM/blob/5eae2f385c014f2250c3130152b6be711f6a3a5a/util/ema.py
"""

from torch.optim import Optimizer


class EMA(Optimizer):
    def __init__(self, optimizer: Optimizer, ema_decay=0.999):
        super().__init__(optimizer.param_groups, {})  # This sets up all internal attributes
        assert 0.0 < ema_decay <= 1.0, "EMA decay must be in (0, 1]"
        self.optimizer = optimizer
        self.ema_decay = ema_decay
        self.num_updates = 0
        self.ema_weights = {}
        self.backup_weights = {}

        # Initialize EMA weights from optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.ema_weights[id(p)] = p.data.clone()

    def step(self, *args, **kwargs):
        """Performs an optimizer step and then updates EMA weights."""
        loss = self.optimizer.step(*args, **kwargs)

        decay = min(self.ema_decay, (1 + self.num_updates) / (10 + self.num_updates))
        self.num_updates += 1

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if id(p) not in self.ema_weights:
                        self.ema_weights[id(p)] = p.data.clone()
                    else:
                        self.ema_weights[id(p)].mul_(decay).add_(
                            p.data, alpha=1 - decay
                        )
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()

    def swap_parameters_with_ema(self, store_params_in_ema=True):
        """Swap model parameters with EMA weights (e.g., for evaluation)."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                param_id = id(p)
                
                # Add this safety check:
                if param_id not in self.ema_weights:
                    # logger.warning(f"Parameter {param_id} not found in EMA weights, initializing...")
                    self.ema_weights[param_id] = p.data.clone()
                    continue

                if store_params_in_ema:
                    self.backup_weights[param_id] = p.data.clone()
                    p.data.copy_(self.ema_weights[param_id])
                else:
                    if param_id in self.backup_weights:
                        p.data.copy_(self.backup_weights[param_id])
                        del self.backup_weights[param_id]

    def state_dict(self):
        """Returns both optimizer state and EMA weights."""
        state = {
            "optimizer": self.optimizer.state_dict(),
            "ema_weights": {k: v.clone() for k, v in self.ema_weights.items()}
        }
        
        return state
    

    def load_state_dict(self, state_dict, device=None):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.ema_weights = {
            k: v.clone().to(device) if device is not None else v.clone()
            for k, v in state_dict["ema_weights"].items()
        }

    def get_ema_model_state_dict(self, model):
        """Convert EMA weights to a model state dict format."""
        ema_state_dict = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_id = id(param)
                if param_id in self.ema_weights:
                    ema_state_dict[name] = self.ema_weights[param_id].clone()
        
        return ema_state_dict