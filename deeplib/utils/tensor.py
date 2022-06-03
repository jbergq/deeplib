import torch


def detach(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()
