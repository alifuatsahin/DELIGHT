
from torch.optim import Optimizer
import torch
import torch.nn as nn

# class EMA(Optimizer):
#     def __init__(self, optimizer: Optimizer, ema_decay=0.999):
#         super().__init__(optimizer.param_groups, {})  # This sets up all internal attributes
#         assert 0.0 < ema_decay <= 1.0, "EMA decay must be in (0, 1]"
#         self.optimizer = optimizer
#         self.ema_decay = ema_decay
#         self.num_updates = 0
#         self.ema_weights = {}
#         self.backup_weights = {}

#         # Initialize EMA weights from optimizer parameters
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 if p.requires_grad:
#                     self.ema_weights[id(p)] = p.data.clone()

#     def step(self, *args, **kwargs):
#         """Performs an optimizer step and then updates EMA weights."""
#         loss = self.optimizer.step(*args, **kwargs)

#         decay = min(self.ema_decay, (1 + self.num_updates) / (10 + self.num_updates))
#         self.num_updates += 1

#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 if p.requires_grad:
#                     if id(p) not in self.ema_weights:
#                         self.ema_weights[id(p)] = p.data.clone()
#                     else:
#                         self.ema_weights[id(p)].mul_(decay).add_(
#                             p.data, alpha=1 - decay
#                         )
#         return loss

#     def zero_grad(self):
#         self.optimizer.zero_grad()

#     def state_dict(self):
#         """Returns both optimizer state and EMA weights."""
#         state = {
#             "optimizer": self.optimizer.state_dict(),
#             "ema_weights": {k: v.clone() for k, v in self.ema_weights.items()}
#         }
        
#         return state

#     def load_state_dict(self, state_dict, device=None):
#         self.optimizer.load_state_dict(state_dict["optimizer"])
#         self.ema_weights = {
#             k: v.clone().to(device) if device is not None else v.clone()
#             for k, v in state_dict["ema_weights"].items()
#         }

#     def get_ema_model_state_dict(self, model):
#         """Convert EMA weights to a model state dict format."""
#         ema_state_dict = {}
        
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param_id = id(param)
#                 if param_id in self.ema_weights:
#                     ema_state_dict[name] = self.ema_weights[param_id].clone()
        
#         return ema_state_dict
    
#     def copy_to(self, model):
#         """Copy EMA weights into the given model."""
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param_id = id(param)
#                 if param_id in self.ema_weights:
#                     param.data.copy_(self.ema_weights[param_id])

#     def store(self, model):
#         """Store current model parameters for later restore."""
#         self.collected_params = [p.data.clone() for p in model.parameters()]

#     def restore(self, model):
#         """Restore model parameters previously stored by `store()`."""
#         for c, p in zip(self.collected_params, model.parameters()):
#             p.data.copy_(c)
#         del self.collected_params


class EMA(nn.Module):
    def __init__(self, model, decay=0.999, use_num_updates=True):
        super().__init__()
        assert 0.0 < decay <= 1.0, "EMA decay must be in (0, 1]"

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_updates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self, model):
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            decay = self.decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_((1.0 - decay) * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)