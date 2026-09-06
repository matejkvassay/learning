import torch
from torch import nn
from mkvlib.models.basic_gpt.multi_head_self_attention import MultiHeadSelfAttention
from mkvlib.models.basic_gpt.feedforward_layer import FeedForwardLayer
from mkvlib.models.basic_gpt.layer_norm import LayerNorm


class DecoderBlock(nn.Module):
    def __init__(self, block_size, n_att_heads, emb_dim, ffw_upscale_factor, dropout):
        super().__init__()
        self.mh_att = MultiHeadSelfAttention(n_att_heads, block_size, emb_dim, dropout=dropout)
        self.ffw = FeedForwardLayer(emb_dim, upscale_factor=ffw_upscale_factor, dropout=dropout)
        self.pre_att_norm = LayerNorm(emb_dim)
        self.pre_ffw_norm = LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: shape B,T,E
        """
        x = self.pre_att_norm(x)
        x = x + self.mh_att(x)
        x = self.pre_ffw_norm(x)
        x = x + self.ffw(x)
        return x
