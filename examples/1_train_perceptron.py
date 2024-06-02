import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from mkvlib.normalize import normalize_zscore
from mkvlib.models import Perceptron
from mkvlib.utils import np_to_torch

'''
CFG
'''
RAND_SEED = 124
N_EPOCHS = 50
LEARNING_RATE = 0.001

'''
DATA PREP
'''

dataset = load_breast_cancer()
dataset_splits = train_test_split(dataset.data, dataset.target, random_state=RAND_SEED)
X_train, X_val, y_train, y_val = np_to_torch(*dataset_splits, dtype=torch.float32)
X_train, mean, std = normalize_zscore(X_train)
X_val, _, _ = normalize_zscore(X_val, mean, std)

'''
TRAIN 
'''

p = Perceptron(n_features=X_train.shape[1], lr=LEARNING_RATE)

for rep in p.train(X_train, y_train, N_EPOCHS, X_val=X_val, y_val=y_val):
    ep = rep['epoch']
    score = rep['score']
    score_val = rep['score_val']
    print(f'\n\n** EPOCH {ep} **')
    print(f"F1 Train: {score}")
    print(f"F1 Val: {score_val}")
