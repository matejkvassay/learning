import torch


def rand_nwp_batch(tokens, batch_size, context_length):
    indices = torch.randint(tokens.shape[0] - context_length - 1, (batch_size,))
    x = torch.stack(tuple(tokens[i:i + context_length] for i in indices))
    y = torch.stack(tuple(tokens[i + 1:i + context_length + 1] for i in indices))
    return x, y
