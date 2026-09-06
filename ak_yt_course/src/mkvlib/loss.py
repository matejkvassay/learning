import torch


def mse(y, y_hat):
    return torch.mean((y - y_hat) ** 2)
