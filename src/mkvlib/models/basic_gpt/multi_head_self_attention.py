import torch
from torch import nn
from mkvlib.models.basic_gpt.self_attention_head import SelfAttentionHead


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, block_size, emb_dim, dropout):
        super().__init__()
        self.att_heads = nn.ModuleList(
            SelfAttentionHead(block_size, emb_dim, emb_dim // n_heads, dropout=dropout) for _ in range(n_heads)
        )
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: shape B, T, E
        :return:  shape B, T, E ; E == n_H x H
        """
        x = tuple(h(x) for h in self.att_heads)  # n_H x B,T,H
        x = torch.cat(x, dim=2)  # B, T, E
        x = self.linear(x)
        x = self.dropout(x)
        return x
