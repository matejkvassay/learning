import torch
from torch import nn
from torch.nn import functional as F
from mkvlib.models.basic_gpt.decoder_block import DecoderBlock
from mkvlib.models.basic_gpt.layer_norm import LayerNorm

'''
Notes
- dropout applied after linear transforms and attention affinity computation
- post-norms replaced with pre-norms  https://arxiv.org/pdf/2002.04745
'''


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layers, block_size, n_att_heads, emb_dim, ffw_upscale_factor,
                 dropout):
        super().__init__()
        self.register_buffer('pos', torch.arange(block_size))
        self.register_buffer('block_size', torch.tensor(block_size))
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        decoders = [
            DecoderBlock(block_size, n_att_heads, emb_dim, ffw_upscale_factor=ffw_upscale_factor, dropout=dropout) for _
            in
            range(n_layers)]
        self.decoders = nn.Sequential(*decoders)
        self.pre_head_norm = LayerNorm(emb_dim)
        self.vocab_clf_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, y=None):
        """
        :param x: shape (B, T)
        :param y: shape (B, T)
        :return: tuple: (tensor shape (B, T, V), loss), loss is None if y not provided
        """
        x = self.tok_emb(x)  # B, T, V -> B, T, E
        x = x + self.pos_emb(self.pos[:x.shape[1]])
        x = self.decoders(x)
        x = self.pre_head_norm(x)
        logits = self.vocab_clf_head(x)  # shape B, T, V

        loss = None
        if y is not None:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)  # reduce dims, cross entropy func requires shape (B,V) for predictions
            y = y.view(B * T)  # ce takes shape (V) for targets
            loss = F.cross_entropy(logits, y)
            logits = logits.view(B, T, V)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """"
        :param idx: (B, T)
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]  # shape (B, <=BLOCK_SIZE)
            # get the predictions
            logits, loss = self(idx_cond)  # logits = shape (B, T, V)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, V) - get probs for last timestep only
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
