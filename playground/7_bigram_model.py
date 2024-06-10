from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import generate_nwp_batch
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
tokens = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(tokens))
tok_train = tokens[:train_size]
tok_test = tokens[train_size:]

assert len(tokenizer) < 125
x_train = torch.tensor(tok_train, dtype=torch.int8)
x_test = torch.tensor(tok_test, dtype=torch.int8)

x, y = generate_nwp_batch(x_train, BATCH_SIZE, CONTEXT_LENGTH)
print(x)
print(y)

#
# class BigramModel(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)
#
#     def forward(self, token_idx, targets):
#         logits = self.emb(token_idx)
#         batch_dim,
