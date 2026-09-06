import torch
from torch import nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, block_size, dim_emb_in, dim_att_head_out, dropout):
        super().__init__()
        self.ln_key = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_query = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_value = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('head_norm', torch.sqrt(torch.tensor(dim_att_head_out, dtype=torch.float32)))

    def forward(self, x):
        """
        :param x: shape B,T,E
        :return: shape B,T,H
        """
        k = self.ln_key(x)  # B, T, E -> B, T, H
        q = self.ln_query(x)  # B, T, E -> B, T, H
        v = self.ln_value(x)  # B, T, E -> B, T, H
        att = q @ k.mT  # B,T,H @ B,H,T ->  B,T,T
        att = att / self.head_norm  # preserves variance of att head outputs to avoid softmax spiking at net init
        # tril only for up to T (if T<max block size)
        att = att.masked_fill(self.tril[:x.shape[1], :x.shape[1]] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # softmax along last dim
        att = self.dropout(att)
        return att @ v  # B,T,T @ B,T,H -> B,T,H
