from torch import nn


class FeedForwardLayer(nn.Module):
    def __init__(self, emb_dim, upscale_factor=4, dropout=0.1):
        super().__init__()
        z = upscale_factor * emb_dim
        self.ffw = nn.Sequential(
            nn.Linear(emb_dim, z),
            nn.ReLU(),
            nn.Linear(z, emb_dim),  # ffw was 4x up-scaled in 2017 paper,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffw(x)
