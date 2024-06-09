from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import re
from unidecode import unidecode
from time import perf_counter

dataset = fetch_20newsgroups()

text = '\n'.join(dataset.data)


def normalize(text):
    text = unidecode(text)
    text = text.strip().lower()
    return text


def pre_split(text, pattern=r'\W+'):
    tokens = re.split(pattern, text)
    for i in range(len(tokens)):
        tokens[i] = tuple(tokens[i])
    tokens = [x for x in tokens if len(x) > 0]
    return tokens


def unique_chars(text):
    char_freq = defaultdict(int)
    for c in text:
        char_freq[c] += 1
    return sorted(set(text)), char_freq


def find_merge_rule(tokens):
    s = perf_counter()
    freq = defaultdict(int)
    for tok in tokens:
        for i in range(0, len(tok) - 1):
            freq[tok[i: i + 2]] += 1
    mr=max(freq.items(), key=lambda x: x[1])[0]
    e = perf_counter()
    print(f'Merge rule search time: {e - s}s')
    return mr


def apply_merge_rule(tokens, merge_rule):
    s = perf_counter()
    for i in range(len(tokens)):
        tok = tokens[i]
        new_tok = []
        last_merged = False
        for j in range(len(tok)):
            if last_merged is True:
                last_merged = False
                continue
            if tuple(tok[j:j + 2]) == merge_rule:
                last_merged = True
                new_tok.append(tok[j] + tok[j + 1])
            else:
                new_tok.append(tok[j])
        tokens[i] = tuple(new_tok)
    e = perf_counter()
    print(f'Merge rule application time: {e - s}s')
    return tokens


text = normalize(text)
tokens = pre_split(text, pattern=' ')
chars, char_freq = unique_chars(text)
merge_rules = []
VOCAB_SIZE = 10000
while len(chars) + len(merge_rules) < VOCAB_SIZE:
    mr = find_merge_rule(tokens)
    merge_rules.append(mr)
    tokens = apply_merge_rule(tokens, mr)
    print(mr)
print(tokens[:1000])
