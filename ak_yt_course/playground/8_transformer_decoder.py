from mkvlib.char_tokenizer import CharTokenizer
from mkvlib.train_utils import rand_nwp_batch
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

'''
Notes
- dropout applied after linear transforms and attention affinity computation
- post-norms replaced with pre-norms  https://arxiv.org/pdf/2002.04745
- dataset used: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
'''


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-05):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x: torch.Tensor):
        """
        :param x: shape B, T, C
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return norm * self.gamma + self.beta


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
        h_out = tuple(h(x) for h in self.att_heads)  # n_H x B,T,H
        x = torch.cat(h_out, dim=-1)  # B, T, E
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, emb_dim, upscale_factor=4, dropout=0.1):
        super().__init__()
        z = upscale_factor * emb_dim
        self.ffw = nn.Sequential(
            nn.Linear(emb_dim, z),
            nn.ReLU(),  # relu was applied on first linear transform only
            nn.Linear(z, emb_dim),  # ffw was 4x up-scaled in 2017 paper
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffw(x)


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
def generate(model, idx, max_new_tokens):
    """"
    :param idx: (B, T)
    """
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx if idx.size(1) <= m.block_size else idx[:, -m.block_size:]  # shape (B, <=BLOCK_SIZE)
        # get the predictions
        logits, loss = model(idx_cond)  # logits = shape (B, T, V)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, V) - get probs for last timestep only
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, V)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx


"""
Training CFG
"""

DATASET_PATH = 'data/skspr.txt'
TRAIN_SPLIT = 0.95
BLOCK_SIZE = 256
BATCH_SIZE = 64
EMB_DIM = 384
N_ATT_HEADS = 6
N_LAYERS = 6
LR = 3e-4
N_TRAINING_BATCHES = 5000
N_EVAL_BATCHES = 20
PRINT_LOSS_AFTER = 100
FFW_UPSCALE_FACTOR = 4
DROPOUT = 0.2

"""
Device detection
"""

if torch.backends.cudnn.is_available():
    device = torch.device("cuda")
    print("using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print('using MPS')
else:
    device = 'cpu'
    print('using CPU')

"""
Dataset reading and tokenization
"""

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer()
indices = tokenizer.fit_transform(text)

train_size = int(TRAIN_SPLIT * len(indices))
x_train = torch.tensor(indices[:train_size])
x_test = torch.tensor(indices[train_size:])

"""
Model training
"""

m = Transformer(vocab_size=len(tokenizer), n_layers=N_LAYERS, block_size=BLOCK_SIZE, n_att_heads=N_ATT_HEADS,
                emb_dim=EMB_DIM, ffw_upscale_factor=FFW_UPSCALE_FACTOR, dropout=DROPOUT)
m = m.to(device)
optim = torch.optim.AdamW(params=m.parameters(), lr=LR)

x_train = x_train.to(device)
for ep in tqdm(range(N_TRAINING_BATCHES)):
    xb, yb = rand_nwp_batch(x_train, BATCH_SIZE, BLOCK_SIZE)
    xb = xb.to(device)
    yb = yb.to(device)
    _, loss = m(xb, yb)
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)

    if (not (ep + 1) % PRINT_LOSS_AFTER) or ep == 0:
        with torch.no_grad():
            m.eval()
            losses_train = []
            losses_val = []

            for _ in range(N_EVAL_BATCHES):
                xb, yb = rand_nwp_batch(x_train, BATCH_SIZE, BLOCK_SIZE)
                xb = xb.to(device)
                yb = yb.to(device)
                x_valb, y_valb = rand_nwp_batch(x_test, BATCH_SIZE, BLOCK_SIZE)
                x_valb = x_valb.to(device)
                y_valb = y_valb.to(device)
                _, train_loss = m.forward(xb, yb)
                _, val_loss = m.forward(x_valb, y_valb)
                losses_val.append(val_loss)
                losses_train.append(train_loss)
            val_loss = torch.mean(torch.Tensor(losses_val))
            train_loss = torch.mean(torch.Tensor(losses_train))
            print(f'ep {ep}: mean cross entropy loss train: {train_loss}')
            print(f'ep {ep}: mean cross entropy loss dev: {val_loss}')
            input = torch.zeros((1, 1), dtype=torch.long).to(device)
            input[0][0] = 11
            print('example generation:')
            res = generate(m, input, 500)
            print(tokenizer.inverse_transform(res[0].tolist()))
            m.train()

"""
Generate example
"""

input = torch.zeros((1, 1), dtype=torch.long).to(device)
input[0][0] = 11
m.eval()
res = generate(m, input, 5000)
print(tokenizer.inverse_transform(res[0].tolist()))
