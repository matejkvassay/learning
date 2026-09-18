from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from train_dataset import TrainDataset


def prepare_dataset(
    data_path: str,
    encoding: str = "gpt2",
    allowed_special: set[str] = {"<|endoftext|>"},
    context_size: int = 128,
    stride: int = 64,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[Tokenizer, DataLoader]:
    with open(data_path, "r") as f:
        text = f.read()
    tokenizer = Tokenizer(encoding=encoding, allowed_special=allowed_special)
    tokens = tokenizer.tokenize(text)
    dataset = TrainDataset(tokens=tokens, context_size=context_size, stride=stride)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
    return tokenizer, loader
