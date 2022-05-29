import torch
import torch.nn as nn
from typing import Callable

from .storage import Storage
from ..utils.tensor import detach
from ..structures import FeatureMap1D, FeatureMap2D, LayerOutput


class ModelAnalyzer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self._activations = Storage()
        self._weights = Storage()

        for name, module in self.model.named_modules():
            module.register_forward_hook(self.store_output_hook(name))
            self._activations[name] = torch.empty(0)

    def store_output_hook(self, module_name: str) -> Callable:
        def fn(layer, input, output):
            output = detach(output).squeeze(0)

            if len(output.shape) == 1:
                self._activations[module_name] = FeatureMap1D(output, module_name)
            elif len(output.shape) == 3:
                self._activations[module_name] = FeatureMap2D(output, module_name)
            else:
                self._activations[module_name] = LayerOutput(output, module_name)

        return fn

    def weights(self):
        return self.model.state_dict()

    def activations(self):
        return self._activations

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
