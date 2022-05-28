import torch
import torch.nn as nn
from typing import Callable


class LogWrapper:
    def __init__(self, model: nn.Module):
        self.model = model
        self._features = {}

        for name, module in self.model._modules.items():
            module.register_forward_hook(self.save_outputs_hook(name))
            self._features[name] = torch.empty(0)

    def save_outputs_hook(self, module_name: str) -> Callable:
        def fn(_, __, output):
            self._features[module_name] = output

        return fn

    def get_log(self):
        return self._features

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
