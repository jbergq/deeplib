import torch
import torch.nn as nn
from typing import Callable

from ..structures import Storage, Activations
from ..utils.tensor import detach
from ..structures import create_activation


class ModelAnalyzer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self._activations = Activations()
        self._weights = Storage()

        for name, module in self.model.named_modules():
            module.register_forward_hook(self.store_output_hook(name))
            self._activations[name] = torch.empty(0)

    def store_output_hook(self, module_name: str) -> Callable:
        def fn(module, input, output):
            output = detach(output).squeeze(0)

            self._activations[module_name] = create_activation(
                output, module_name, module
            )

        return fn

    def weights(self):
        return self.model.state_dict()

    def activations(self):
        return self._activations

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
