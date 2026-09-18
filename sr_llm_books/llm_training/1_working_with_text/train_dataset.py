import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, tokens: list[int], context_size: int, stride: int):
        if stride > context_size or stride < 1:
            raise ValueError(
                "Stride has to be between 1 and context_size. "
                "1 = maximum overlap, context_size = minimum overlap."
            )
        self.tokens = tokens
        self.context_size = context_size
        self.stride = stride
        self.sample_indices = list(range(0, len(tokens) - context_size, stride))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.sample_indices[index]
        x = self.tokens[i : self.context_size + i]
        y = self.tokens[i + 1 : self.context_size + i + 1]
        return (torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64))

    def __len__(self) -> int:
        return len(self.sample_indices)
