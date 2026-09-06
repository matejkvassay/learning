import torch


def np_to_torch(*arrays, dtype=torch.float32):
    return (torch.from_numpy(x).to(dtype) for x in arrays)
