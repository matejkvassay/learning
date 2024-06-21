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
N_EPOCHS = 1000

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer()
indices = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(indices))
x_train = torch.tensor(indices[:train_size])
x_test = torch.tensor(indices[train_size:])

N_EMB = 16


class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim):
        super().__init__()
        # vars
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # layers
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = nn.Embedding(self.block_size, self.emb_dim)
        self.head = nn.Linear(self.emb_dim, self.vocab_size)

        # buffers
        pos = torch.arange(self.block_size)  # positions 0 - block_size
        tril = torch.tril(torch.ones(self.block_size, self.block_size))
        tril = tril.masked_fill(tril == 0, float('-inf'))
        self.register_buffer('pos', pos)
        self.register_buffer('tril', tril)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        x = self.tok_emb(x)
        x = x + self.pos_emb(self.pos)  # shape B, T, C
        scores = self.head(x)
        loss = None
        if y is None:
            B, T, V = scores.shape
            scores = scores.view(B * T, V) # ce takes shape (B,C)
            y = y.view(B * T) # ce takes shape (C)
            loss = F.cross_entropy(scores, y)  # takes shape (B, |V|)
        return scores, loss


t = Transformer(vocab_size=len(tokenizer), block_size=BLOCK_SIZE, emb_dim=EMB_DIM)

for ep in N_EPOCHS:
    t.zero_grad(set_to_none=True)
    x, y = rand_nwp_batch(x_train, BATCH_SIZE, BLOCK_SIZE)
    y_hat = t(x)
