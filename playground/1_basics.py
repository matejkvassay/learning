import torch
import numpy as np

# https://stackoverflow.com/questions/73924697/whats-the-difference-between-torch-mm-torch-matmul-and-torch-mul

DATA = [
    [
        [.4, .5, .1],
        [1., .9, .2]
    ],
    [
        [.4, .8, .7],
        [.2, 0., .2]
    ],
    [
        [.7, .8, .7],
        [.3, .6, .2]
    ],
    [
        [.1, .5, .1],
        [.1, .2, .1]
    ]
]

'''
Basic
'''
a = torch.tensor(DATA, dtype=torch.float16)

print(f'Shape: {a.shape}')
print(f'Rank: {a.ndim}')
print(f'Dtype: {a.dtype}')

'''
From np array
'''
b = np.array(DATA, dtype=np.float16)
print(f'Type of array b: {type(b)}')
b = torch.from_numpy(b)  # uses the same allocated memory as the original  np array
print(f'Type of tensor b: {type(b)}')
print(f'Dtype of tensor b: {b.dtype}')  # takes type from numpy

'''
Type conversion
'''
c = b.to(torch.float32)
print(f'Dtype after float8 conversion: {c.dtype}')
print(f'Tensor device: {c.device}')

'''
Reshape
'''
d = torch.tensor(DATA, dtype=torch.float16)
print(f'Original shape: {d.shape}')
print(f'Single row: {d.view(1, -1).shape}')  # -1 determines shape automatically
print(f'Single column: {d.view(-1, 1).shape}')
print(f'Matrix with 2 columns: {d.view(-1, 2).shape}')
print(f'Matrix with 3 rows: {d.view(3, -1).shape}')

'''
Transpose
'''
mtx_2d = d[0]
mtx_3d = d
print(f'Original shape: {mtx_2d.shape} ')
print(f'Transposed: {mtx_2d.T.shape}')
print(f'Original shape: {mtx_3d.shape}')
print(f'Transposed: {mtx_3d.mT.shape}')

'''
Weighted sum of vectors (dot product)
'''
x = torch.tensor([.2, .3, .4, .6])
w = torch.tensor([.4, .4, .1, .1])
print(f'Vectors: x={x}, w={w}')
print(f'Weighted sum (dot product): {x.dot(w)}')
assert x.dot(w) == .2 * .4 + .3 * .4 + .4 * .1 + .6 * .1

'''
Matrix multiplication (Weighted sum for batches of training examples)
'''
x = torch.tensor(DATA[0])
v = torch.tensor(DATA[2][1])
w = torch.tensor(DATA[1])
print(f'Matrix X:{x.shape}, vector v: {v.shape}, weight matrix W: {w.shape}')
print(f'Multiply matrix {x.shape} by vector {v.shape}: {x.matmul(v).shape}')
print(f'Multiply matrix {x.shape} by matrix {x.shape}: {x.matmul(w.T).shape}')

'''
Broadcasting 
'''
v = torch.tensor(DATA[2][1])
x = .3
print(f'v={v}, x={x}, v+x = {v + x}')

x = torch.tensor([[1, 2, 4, 9], [4, 5, 6, 0]])
v = torch.tensor([0, 5, 10, 15])
print(f'X = {x}, v = {v} X+v = {x + v}')

'''
Triangular matrix column averaging
'''
a = torch.tril(torch.ones((3, 3)))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print(a)
print(b)
print(c)

'''
Triangular normalized mtx with softmax 
'''
import torch
from torch.nn import functional as F

B, T, C = (4, 8, 2)
x = torch.randn(B, T, C)
tril = torch.tril(torch.ones(T, T))
w = torch.zeros(T, T)
w = w.masked_fill(tril == 0, float('-inf'))
w = F.softmax(w, dim=1)  # trick - normalization function, divides 1nes into equal partials
z = w @ x
print(z)
print(w.shape, x.shape)
print(z.shape)

'''
Pytorch MPS
'''
import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")

'''
Transpose with axes spec
'''
a = torch.zeros((8, 16, 32))
print(a.shape)
print(a.T.shape)  # soon to be deprecated
print(a.mT.shape)  # alternative to transpose batched matrices
print(a.transpose(-1, -2).shape)
print(a.transpose(-2, -1).shape)  # transpose is invariant to order of axis params

'''
stack method to create batches 
'''
tensors = tuple(torch.zeros(32, 16) for _ in range(4))  # 32 features, 16 channels embedding
stacked = torch.stack(tensors)
print(stacked.shape)  # creates batch 4, 32, 16

'''
concatenate with cat() along embedding dimension (for multi-head attention concat)
'''
tensors = tuple(torch.zeros(4, 32, 16) for _ in range(4))  # 4 x tensor 8 batch size, 32 features, 16 channels embedding
concatenated = torch.cat(tensors, dim=2)
print(concatenated.shape)  # result 8, 32, 64 - embeddings concatenated to 4 x 16 channel
