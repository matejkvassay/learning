import torch
from sklearn.metrics import f1_score


class Perceptron:
    def __init__(self, n_features=None, lr=0.001):
        self.n_features = n_features
        self.w = torch.rand(n_features)
        self.b = torch.tensor(0.0)
        self.lr = torch.tensor(lr)

    def forward(self, x):
        xw = x.matmul(self.w.T) + self.b
        return xw.ge(0).to(torch.float32)

    def _update_weights(self, x, y):
        y_hat = self.forward(x)
        e = y - y_hat
        self.b = self.b + self.lr.mul(e)
        self.w = self.w + self.lr.mul(e).mul(x)

    def _score(self, score_func, X, y):
        y_hat = self.forward(X)
        return score_func(y, y_hat)

    def train(self, X, y, n_epochs, score_func=f1_score, X_val=None, y_val=None):
        for ep in range(n_epochs):
            # fit
            shuffled_indices = torch.randperm(X.shape[0])
            for i in shuffled_indices:
                self._update_weights(X[i], y[i])
            # eval
            score = self._score(score_func, X, y)
            score_val = None
            if X_val is not None and y_val is not None:
                score_val = self._score(score_func, X_val, y_val)

            yield {"epoch": ep + 1, "score": score, "score_val": score_val}
