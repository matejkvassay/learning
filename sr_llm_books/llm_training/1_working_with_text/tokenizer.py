from typing import Iterable

import tiktoken


class Tokenizer:
    def __init__(self, encoding: str = "gpt2", allowed_special={"<|endoftext|>"}):
        self.allowed_special = allowed_special
        self.tokenizer = tiktoken.get_encoding(encoding)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special=self.allowed_special)

    def decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(tokens)
