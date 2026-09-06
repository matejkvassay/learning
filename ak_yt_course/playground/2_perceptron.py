import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

RAND_SEED = 124
NORMALIZATION = 0  # 0 = column -wise, 1 = row-wise
N_EPOCHS = 100
LEARNING_RATE = 0.001

'''
Data Load
'''

dataset = load_breast_cancer()
features = dataset.data
targets = dataset.target
X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=RAND_SEED)

X_train = torch.from_numpy(X_train).to(torch.float32)
y_train = torch.from_numpy(y_train).to(torch.float32)
X_test = torch.from_numpy(X_test).to(torch.float32)
y_test = torch.from_numpy(y_test).to(torch.float32)

'''
Func/Cls definitions
'''


def column_wise_norm(x_train, x_pred):
    tol = 0.000000001
    norm_stdev = x_train.std(dim=0) + tol
    norm_mean = x_train.mean(dim=0)
    x_train = (x_train - norm_mean).div(norm_stdev)
    x_pred = (x_pred - norm_mean).div(norm_stdev)
    return x_train, x_pred


def row_wise_norm(x):
    tol = torch.full((x.shape[0],), 0.000000001)
    print(f'tol shape: {tol.shape}')
    norm = torch.norm(x, dim=1)
    print(f'norm shape: {norm.shape}')
    return x.T.div(torch.maximum(norm, tol)).T


class Perceptron:
    def __init__(self, n_features, lr=0.001):
        self.n_features = n_features
        self.w = torch.rand(n_features)
        self.b = torch.tensor(0.0)
        self.lr = torch.tensor(lr)

    def forward(self, x):
        xw = x.matmul(self.w.T) + self.b
        return xw.ge(0).to(torch.float32)

    def train(self, x, y):
        y_hat = self.forward(x)
        e = y - y_hat
        self.b = self.b + self.lr.mul(e)
        self.w = self.w + self.lr.mul(e).mul(x)


def column_wise_norm(x_train, x_pred):
    tol = 0.000000001
    norm_stdev = x_train.std(dim=0) + tol
    norm_mean = x_train.mean(dim=0)
    x_train = (x_train - norm_mean).div(norm_stdev)
    x_pred = (x_pred - norm_mean).div(norm_stdev)
    return x_train, x_pred


def row_wise_norm(x):
    tol = torch.full((x.shape[0],), 0.000000001)
    print(f'tol shape: {tol.shape}')
    norm = torch.norm(x, dim=1)
    print(f'norm shape: {norm.shape}')
    return x.T.div(torch.maximum(norm, tol)).T


'''
Data norm
'''

if NORMALIZATION == 1:
    X_train = row_wise_norm(X_train)
    X_test = row_wise_norm(X_test)
elif NORMALIZATION == 0:
    X_train, X_test = column_wise_norm(X_train, X_test)
else:
    raise ValueError('Unknown data normalization setting, has to be 0 or 1.')

'''
Train
'''
perceptron = Perceptron(n_features=X_train.shape[1], lr=LEARNING_RATE)

for ep in range(N_EPOCHS):
    indices = torch.randperm(X_train.shape[0])
    for i in indices:
        perceptron.train(X_train[i], y_train[i])
    y_hat_train = perceptron.forward(X_train)
    y_hat_test = perceptron.forward(X_test)
    print(f"train acc epoch {ep + 1} = {f1_score(y_train, y_hat_train)}")
    print(f"test acc epoch {ep + 1} = {f1_score(y_test, y_hat_test)}")
