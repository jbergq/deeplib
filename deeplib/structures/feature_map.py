import math

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def create_activation(output, name, module):
    if len(output.shape) == 1:
        return FeatureMap1D(output, name)
    elif len(output.shape) == 3:
        return FeatureMap2D(output, name)
    else:
        return LayerOutput(output, name)


def put_text(img, text):
    return cv2.putText(
        img,
        text,
        (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


class LayerOutput:
    def __init__(self, data, name="unnamed_output") -> None:
        self.data = data
        self.name = name


class FeatureMap1D(LayerOutput):
    def __init__(self, data, name=None) -> None:
        super().__init__(name)
        self.data = data

    def image(self):
        grid_squares = [np.ones((64, 64)) * d.item() for d in self.data]
        grid_squares = [put_text(gs, str(idx)) for idx, gs in enumerate(grid_squares)]

        grid = make_grid(
            [torch.tensor(d).unsqueeze(0) for d in grid_squares],
            nrow=math.ceil(math.sqrt(self.data.shape[0])),
        )

        if grid.shape[0] == 1 or grid.shape[0] == 3:
            grid = grid.permute(1, 2, 0)

        return grid


class FeatureMap2D(LayerOutput):
    def __init__(self, data, name=None) -> None:
        super().__init__(data, name)

    def image(self):
        grid = make_grid(
            [d.unsqueeze(0) for d in self.data],
            nrow=math.ceil(math.sqrt(self.data.shape[0])),
        )

        if grid.shape[0] == 1 or grid.shape[0] == 3:
            grid = grid.permute(1, 2, 0)

        return grid
