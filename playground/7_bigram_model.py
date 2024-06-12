from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import rand_nwp_batch
import torch
from torch import nn
from torch.nn import functional as F

DATASET_PATH = '/Users/A109096228/data/sample_dataset.txt'
OOV_CHAR = ' '
TRAIN_SPLIT = 0.95
CONTEXT_LENGTH = 4
BATCH_SIZE = 32
N_EPOCHS = 1000

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer(' ')
indices = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(indices))
x_train = torch.tensor(indices[:train_size])
x_test = torch.tensor(indices[train_size:])


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets=None):
        logits = self.emb(idx)  # dims: (Batch size , Time (context) 8, Channel (vocab size))
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view((B * T, C))
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self(idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx


m = BigramModel(len(tokenizer))
optim = torch.optim.AdamW(params=m.parameters(), lr=0.01)
for ep in range(N_EPOCHS):
    xb, yb = rand_nwp_batch(x_train, BATCH_SIZE, CONTEXT_LENGTH)
    _, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    print(f'ep: {ep} loss:{loss.item()}')

m.eval()
sample = torch.zeros((1, 1), dtype=torch.long)
res = m.generate(sample, 500)
print(res[0])
print(tokenizer.inverse_transform(res[0].tolist()))
