import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from mkvlib.normalize import normalize_zscore
from mkvlib.models.logistic_regression import LogisticRegression

RAND_SEED = 124
NORMALIZATION = 0  # 0 = column -wise, 1 = row-wise
N_EPOCHS = 100
LEARNING_RATE = 0.001

dataset = load_breast_cancer()
features = dataset.data
targets = dataset.target
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=RAND_SEED)

X_train = torch.from_numpy(X_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.float32)

X_train, mean, std = normalize_zscore(X_train)
X_test, _, _ = normalize_zscore(X_test, mean, std)

model = LogisticRegression(X_train.shape[1], lr=LEARNING_RATE)

for ep in range(N_EPOCHS):
    loss = model.train(X_train, y_train)
    print(loss.item())
