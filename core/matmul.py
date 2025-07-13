# core/matmul.py

from core.autograd import Tensor

def matmul(a, b):
    result = []
    for i in range(len(a.data)):
        row = []
        for j in range(len(b[0])):
            val = sum(a.data[i][k] * b[k][j] for k in range(len(b)))
            row.append(val)
        result.append(row)
    return Tensor(result, requires_grad=a.requires_grad or b.requires_grad)

def transpose(tensor):
    transposed = [[tensor.data[j][i] for j in range(len(tensor.data))]
                  for i in range(len(tensor.data[0]))]
    return Tensor(transposed, requires_grad=tensor.requires_grad)

