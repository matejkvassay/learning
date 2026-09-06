from mkvlib.scalar_autograd import Scalar

w = Scalar(0.55)
x = Scalar(0.1)
b = Scalar(0.8)
n = Scalar(0.2)

L = (w * x + b) / n

L.backprop_lets_go()

print(w.grad)
print(x.grad)
print(n.grad)
print(L.grad)
print(L.inputs)
