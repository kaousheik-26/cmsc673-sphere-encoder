import copy
import math
import torch
import torch.nn as nn

"""
classes
"""


def _named_tensors(pairs):
    """
    pairs : iterable of (name, tensor), e.g. module.named_parameters()
    out   : dict {name: tensor}
    """
    return {
        n.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", ""): t
        for n, t in pairs
    }


class ModuleEMA(nn.Module):
    """
    exponential moving average of model weights, wrapped as a native nn.Module.
    this allows seamless forward passes while maintaining the compute graph

    model : the live model, deep-copied to hold the averaged weights
    decay : ema decay per step
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.decay = decay

        # duplicate the entire model
        self.module = copy.deepcopy(model)
        self.module.eval()

        # freeze the EMA model's parameters,
        # but allows gradients to flow THROUGH the module to the inputs.
        for param in self.module.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def step(self, active_model: nn.Module):
        # handle DDP if used
        if isinstance(active_model, torch.nn.parallel.DistributedDataParallel):
            active_model = active_model.module

        active_params = _named_tensors(active_model.named_parameters())
        ema_params = _named_tensors(self.module.named_parameters())

        for name, active_param in active_params.items():
            if name in ema_params:
                if active_param.requires_grad:
                    ema_params[name].mul_(self.decay).add_(
                        active_param.detach(), alpha=1 - self.decay
                    )
                else:
                    ema_params[name].copy_(active_param.detach())

        # copy buffers
        active_buffers = _named_tensors(active_model.named_buffers())
        ema_buffers = _named_tensors(self.module.named_buffers())
        for name, active_buffer in active_buffers.items():
            if name in ema_buffers:
                ema_buffers[name].copy_(active_buffer.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        write the ema weights into `model` in place (the inverse of step()).
        used by load_ckpt(override_model_with_ema=True) so sampling / eval run
        on the ema weights

        model : the model to overwrite, optionally wrapped in ddp
        """
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        ema_params = _named_tensors(self.module.named_parameters())
        for name, param in _named_tensors(model.named_parameters()).items():
            if name in ema_params:
                param.copy_(ema_params[name].detach())

        ema_buffers = _named_tensors(self.module.named_buffers())
        for name, buf in _named_tensors(model.named_buffers()).items():
            if name in ema_buffers:
                buf.copy_(ema_buffers[name].detach())

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.module.load_state_dict(state_dict, *args, **kwargs)


"""
functions
"""


def get_ema_decay(current_step, total_steps, start_decay=0.99, end_decay=0.9999):
    """
    cosine schedule for the ema decay from start_decay to end_decay over
    total_steps

    current_step : current optimizer step
    total_steps  : total number of optimizer steps
    start_decay  : decay at step 0
    end_decay    : decay at total_steps
    out          : decay for the current step
    """
    progress = current_step / total_steps
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    decay = end_decay - (end_decay - start_decay) * cosine_val
    return decay
