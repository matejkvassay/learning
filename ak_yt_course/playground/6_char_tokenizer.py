from mkvlib.char_tokenizer import CharTokenizer

DATASET_PATH = '/Users/A109096228/data/sample_dataset.txt'
OOV_CHAR = ' '

with open(DATASET_PATH, 'r') as f:
    text = f.read().strip()

tokenizer = CharTokenizer(' ')
tokens = tokenizer.fit_transform(text)

print('DEC __________')
print(tokenizer.dec_map)
print('ENC __________')
print(tokenizer.enc_map)
print('OOV __________')
print(tokenizer.dec_oov_idx)
print('EXAMPLE __________')
q = 'hello there!#%#^&$@^%O)@P{"#Q:'
print(f'Original str: {q}')
enc = tokenizer.transform(q)
dec = tokenizer.inverse_transform(enc)
print(f'Encoded: {enc}')
print(f'Decoded: {dec}')
