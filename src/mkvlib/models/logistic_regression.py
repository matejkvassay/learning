from mkvlib.activation import sigmoid
from mkvlib.layer import linear
from mkvlib.loss import mse
import torch


class LogisticRegression:
    def __init__(self, n_features, n_weights, lr=0.0001, loss_f=mse):
        self.W = torch.rand((n_features, n_weights), requires_grad=True)
        self.b = torch.rand(n_features, requires_grad=True)
        self.loss_f = loss_f
        self.lr = lr

    def _sgd(self, loss, *weights):
        loss.backward_fn()
        print(loss.grad_fn)
        print(self.W)
        print(self.b)
        for w in weights:
            print(f"grad:{w.grad}")
        weights_updated = tuple(w - w.grad * self.lr for w in weights)
        self.W.grad = None
        self.b.grad = None
        return weights_updated

    def forward(self, X):
        return sigmoid(linear(X, self.W, self.b))

    def train(self, X, y):
        y_hat = self.forward(X)
        loss = self.loss_f(y, y_hat)
        self.W, self.b = self._sgd(loss, self.W, self.b)
        return loss
