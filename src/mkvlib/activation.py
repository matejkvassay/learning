import torch


def sigmoid(Z: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-Z))
