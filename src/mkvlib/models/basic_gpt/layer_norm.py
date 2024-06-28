import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x: torch.Tensor):
        """
        :param x: shape B, T, C
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        return ((x - mean) / torch.sqrt(var + self.eps)) * self.gamma + self.beta
