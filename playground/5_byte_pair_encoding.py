from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import re

dataset = fetch_20newsgroups()

text = '\n'.join(dataset.data)


def normalize(text):
    return text.strip().lower()


def pre_split(text, pattern=r'\W+'):
    tokens = re.split(pattern, text)
    for i in range(len(tokens)):
        tokens[i] = tuple(tokens[i])
    return tokens


def unique_chars(text):
    return sorted(set(text))


def find_merge_rule(tokens):
    freq = defaultdict(int)
    for tok in tokens:
        for i in range(0, len(tok) - 1):
            freq[tok[i: i + 2]] += 1
    return max(freq.items(), key=lambda x: x[1])[0]


def apply_merge_rule(tokens, merge_rule):
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
    return tokens


tokens = pre_split(normalize(text))
chars = unique_chars(text)
merge_rules = set()

VOCAB_SIZE = 10000
while len(chars)+len(merge_rules)<VOCAB_SIZE:
    mr = find_merge_rule(tokens)
    merge_rules.add(mr)
    tokens = apply_merge_rule(tokens, mr)
    print(mr)
print(tokens[:100])

