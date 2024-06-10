import torch


def generate_nwp_batch(tokens, batch_size, context_length):
    x = []
    y = []
    for _ in range(batch_size):
        idx_start = torch.randint(tokens.shape[0] - context_length - 1, (1, 1))
        x.append(tokens[idx_start:idx_start + context_length])
        y.append(tokens[idx_start + 1:idx_start + context_length + 1])
    return torch.stack(x), torch.stack(y)
