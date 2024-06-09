from mkvlib.activation import sigmoid
from mkvlib.loss import mse
import torch


class LogisticRegression:
    def __init__(self, n_features, lr=0.0001, loss_f=mse):
        self.w = torch.rand(n_features, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)
        self.loss_f = loss_f
        self.lr = lr

    def forward(self, X):
        return sigmoid(X.matmul(self.w.T) + self.b)

    def train(self, X, y):
        y_hat = self.forward(X)
        print(y_hat.shape)
        loss = self.loss_f(y, y_hat)
        loss.backward()
        with torch.no_grad():
            self.w -= self.w.grad
            self.b -= self.b.grad
            self.w.grad.zero_()
            self.b.grad.zero_()
        return loss
