from collections import defaultdict
from typing import Iterable


class CharTokenizer:

    def __init__(self, oov_char=' '):
        self.enc_map = defaultdict(self._get_new_idx)
        self.dec_map = None
        self.enc_idx_next = 0
        self.oov_char = oov_char

    def fit_transform(self, text: str) -> tuple[int, ...]:
        if self.enc_map is dict:
            raise RuntimeError('Cannot fit CharTokenizer twice, initialize new instance.')
        char_indices = self._tokenize(text)
        self._freeze()
        return char_indices

    def transform(self, text: str) -> tuple[int, ...]:
        if isinstance(self.enc_map, defaultdict):
            raise RuntimeError('You must first fit tokenizer using fit_transform() method.')
        return self._tokenize(text)

    def inverse_transform(self, token_ids: Iterable[int]) -> str:
        return ''.join(map(self._int_to_char, token_ids))

    def _freeze(self):
        if self.oov_char not in self.enc_map.keys():
            self._char_to_int(self.oov_char)
        self.enc_map = dict(self.enc_map)
        self.dec_map = {v: k for (k, v) in self.enc_map.items()}
        self.dec_oov_idx = self.enc_map[self.oov_char]

    def _get_new_idx(self):
        self.enc_idx_next += 1
        return self.enc_idx_next - 1

    def _char_to_int(self, char: str) -> int:
        try:
            return self.enc_map[char]
        except KeyError:
            return self.enc_map[self.oov_char]

    def _int_to_char(self, char_index: int) -> str:
        try:
            return self.dec_map[char_index]
        except KeyError:
            return self.dec_map[self.enc_map[self.oov_char]]

    def _tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self._char_to_int(c) for c in text)
