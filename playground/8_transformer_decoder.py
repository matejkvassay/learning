from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import rand_nwp_batch
import torch
from torch import nn
from torch.nn import functional as F

DATASET_PATH = '/Users/matej/data/learning/skspr.txt'
TRAIN_SPLIT = 0.95
BLOCK_SIZE = 8
BATCH_SIZE = 32
EMB_DIM = 128
N_ATT_HEADS = 16
LR = 0.001
N_EPOCHS = 10000
PRINT_LOSS_AFTER = 100

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
    print('using MPS')
else:
    device = 'cpu'


class SelfAttentionHead(nn.Module):
    def __init__(self, block_size, dim_emb_in, dim_att_head_out):
        super().__init__()
        self.ln_key = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_query = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)
        self.ln_value = nn.Linear(dim_emb_in, dim_att_head_out, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('head_norm', 1 / torch.sqrt(torch.tensor(dim_att_head_out)))

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
        att = att.masked_fill(self.tril == 0, float('-inf'))
        att = F.softmax(att, dim=1)  # softmax along T dimension => for all tokens in context sum of logits == 1
        return att @ v  # B,T,T @ B,T,H -> B,T,H


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, block_size, emb_dim):
        super().__init__()
        self.att_heads = tuple(
            SelfAttentionHead(block_size, emb_dim, emb_dim // n_heads) for _ in range(n_heads)
        )

    def forward(self, x):
        """
        :param x: shape B, T, E
        :return:  shape B, T, E ; E == n_H x H
        """
        head_outputs = tuple(h(x) for h in self.att_heads)  # n_H x B,T,H
        return torch.cat(head_outputs, dim=2)  # B, T, E


class TransformerDecoderBlock(nn.Module):
    def __init__(self, vocab_size, block_size, n_att_heads, emb_dim):
        super().__init__()
        # layers
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.vocab_clf_head = nn.Linear(emb_dim, vocab_size)
        self.mh_att = MultiHeadSelfAttention(n_att_heads, block_size, emb_dim)

        self.register_buffer('pos', torch.arange(block_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        """
        :param x: shape B,T
        :param y: shape B,T
        :return: predicted token logits (shape B,T,V), loss (if param y not None)
        """
        x = self.tok_emb(x)  # B, T, V -> B, T, E
        x = x + self.pos_emb(self.pos)
        x = self.mh_att(x)
        y_hat = self.vocab_clf_head(x)  # shape B, T, V

        loss = None
        if y is not None:
            B, T, V = y_hat.shape
            y_hat = y_hat.view(B * T, V)  # reduce dims, cross entropy func requires shape (B,V) for predictions
            y = y.view(B * T)  # ce takes shape (V) for targets
            loss = F.cross_entropy(y_hat, y)
            y_hat = y_hat.view(B, T, V)
        return y_hat, loss


m = TransformerDecoderBlock(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, n_att_heads=N_ATT_HEADS, emb_dim=EMB_DIM)
optim = torch.optim.AdamW(params=m.parameters(), lr=LR)

for ep in range(N_EPOCHS):
    xb, yb = rand_nwp_batch(x_train, BATCH_SIZE, BLOCK_SIZE)
    _, loss = m(xb, yb)
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    if not ep % PRINT_LOSS_AFTER:
        print(f'ep {ep}: cross entropy loss train: {loss}')
