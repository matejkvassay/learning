import math

"""
Inspired by: 
https://github.com/karpathy/micrograd
"""


class Scalar:
    def __init__(self, val, inputs=(), op='na'):
        self.val = val
        self.inputs = inputs
        self.op = op
        self.grad = 0.0
        self.backward_fn = lambda: None

    def backprop_lets_go(self):
        self.zero_grad_baby()
        queue = self.bfs_queue()

        self.grad = 1.0
        while len(queue) > 0:
            x = queue.pop()
            x.backward_fn()

    def bfs_queue(self):
        visited = set()
        queue = list()

        def bfs_dag(s: Scalar):
            if s not in visited:
                visited.add(s)
                for c in s.inputs:
                    bfs_dag(c)
            queue.append(s)

        bfs_dag(self)
        return queue

    def zero_grad_baby(self):
        self.grad = 0.0
        for child in self.inputs:
            child.zero_grad_baby()

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        out = Scalar(self.val + other.val, (self, other))

        def b():
            self.grad += out.grad
            other.grad += out.grad

        out.backward_fn = b
        return out

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        out = Scalar(self.val * other.val, (self, other), '*')

        def b():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad

        out.backward_fn = b
        return out

    def __pow__(self, power, modulo=None):
        assert isinstance(power, (int, float))
        out = Scalar(self.val ** power, (self,), f'**{power}')

        def b():
            self.grad += out.grad * (power * (self.val ** (power - 1)))

        out.backward_fn = b
        return out

    def exp(self):
        out = Scalar(math.exp(self.val), (self,), 'exp')

        def b():
            self.grad += out.val * out.grad

        out.backward_fn = b
        return out

    def tanh(self):
        x = (2 * self).exp()
        return (x - 1) / (x + 1)

    def relu(self):
        val = 0.0 if self.val < 0 else self.val
        out = Scalar(val, (self,), 'relu')

        def b():
            self.grad += int(out.val > 0) * out.grad

        out.backward_fn = b
        return out

    def __neg__(self):
        return Scalar(-1) * self

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        other = Scalar(other)
        return other + self

    def __rsub__(self, other):
        other = Scalar(other)
        return other - self

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rmul__(self, other):
        other = Scalar(other)
        return other.__mul__(self)

    def __repr__(self):
        return f'{round(self.val, 4)}|{round(self.grad, 4)}|{self.op}'
