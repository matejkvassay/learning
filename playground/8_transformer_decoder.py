from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import rand_nwp_batch
import torch
from torch import nn
from torch.nn import functional as F

DATASET_PATH = '/Users/A109096228/data/sample_dataset.txt'
TRAIN_SPLIT = 0.95
BLOCK_SIZE = 8
BATCH_SIZE = 4
EMB_DIM = 2
N_ATT_HEADS = 4
N_EPOCHS = 1000

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer()
indices = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(indices))
x_train = torch.tensor(indices[:train_size])
x_test = torch.tensor(indices[train_size:])

N_EMB = 16

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    print("MPS device not found.")


class SelfAttentionHead(nn.Module):
    def __init__(self, block_size, dim_emb_in, dim_att_head_out, masked=True):
        super().__init__()
        self.masked = masked

        # K,Q,V linear transform, usually without bias
        self.ln_key = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_query = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_value = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)

        # buffers
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('head_norm', 1 / torch.sqrt(dim_att_head_out))

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
        if self.masked is True:
            att = att.masked_fill(self.tril == 0, float('-inf'))
        att = F.softmax(att, dim=1)  # softmax along T dimension => for all tokens in context sum of logits == 1
        return att @ v  # B,T,T @ B,T,H -> B,T,H


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, block_size, emb_dim, masked=True):
        super().__init__()
        self.att_heads = tuple(
            SelfAttentionHead(block_size, emb_dim, emb_dim / n_heads, masked=masked) for _ in range(n_heads)
        )

    def forward(self, x):
        """
        :param x: shape B, T, E
        :return:  shape B, T, E, E == n_H x H
        """
        head_outputs = tuple(h(x) for h in self.att_heads)  # n_H x B,T,H
        return torch.cat(head_outputs, dim=2)  # B, T, E


class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, block_size, n_att_heads, emb_dim):
        super().__init__()
        # vars
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # layers
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.block_size, self.emb_dim)
        self.clf_head = nn.Linear(self.emb_dim, self.vocab_size)
        self.mh_att = MultiHeadSelfAttention(n_att_heads, block_size, emb_dim)

        self.register_buffer('pos', torch.arange(self.block_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        x = self.tok_emb(x)  # B, T, V -> B, T, E
        x = x + self.pos_emb(self.pos)  # B, T, E
        x = self.mh_att(x)
        logits = self.clf_head(x)  # B, T, E

        loss = None
        if y is not None:
            B, T, E = logits.shape
            scores = logits.view(B * T, E)  # ce takes shape (B,C)
            y = y.view(B * T)  # ce takes shape (C)
            loss = F.cross_entropy(scores, y)  # takes shape (B, |V|)
        return scores, loss


t = TransformerBlock(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, n_att_heads=N_ATT_HEADS, emb_dim=EMB_DIM)

for ep in N_EPOCHS:
    t.zero_grad(set_to_none=True)
    x, y = rand_nwp_batch(x_train, BATCH_SIZE, BLOCK_SIZE)
    y_hat = t(x)
