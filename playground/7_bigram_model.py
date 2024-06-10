from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import rand_nwp_batch
import torch
from torch import nn
from torch.nn import functional as F

DATASET_PATH = '/Users/A109096228/data/sample_dataset.txt'
OOV_CHAR = ' '
TRAIN_SPLIT = 0.95
CONTEXT_LENGTH = 8
BATCH_SIZE = 4

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer(' ')
indices = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(indices))
x_train = torch.tensor(indices[:train_size])
x_test = torch.tensor(indices[train_size:])

x, y = rand_nwp_batch(x_train, BATCH_SIZE, CONTEXT_LENGTH)


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, X, Y=None):
        logits = self.emb(X)  # dims: (Batch size , Time (context) 8, Channel (vocab size))
        if Y is None:
            return logits,

        B, T, C = logits.shape
        logits = logits.view((B * T, C))
        Y = Y.view(-1)
        loss = F.cross_entropy(logits, Y)
        return logits, loss

    def generate(self, x, max_tokens):
        for _ in range()


m = BigramModel(len(tokenizer))
print(m(x, y))

print(m.parameters())