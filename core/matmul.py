# matmul.py — matrix multiplication with gradients

from core.autograd import Tensor

def matmul(a: Tensor, b: Tensor) -> Tensor:
    out_data = []
    for i in range(len(a.data)):
        row = []
        for j in range(len(b.data[0])):
            val = sum(a.data[i][k] * b.data[k][j] for k in range(len(b.data)))
            row.append(val)
        out_data.append(row)

    out = Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = [[1 for _ in row] for row in a.data]
        if b.requires_grad:
            b.grad = [[1 for _ in row] for row in b.data]
    out._backward = _backward
    out._prev = {a, b}
    return out

def transpose(t: Tensor) -> Tensor:
    transposed = list(map(list, zip(*t.data)))
    return Tensor(transposed, requires_grad=t.requires_grad)

