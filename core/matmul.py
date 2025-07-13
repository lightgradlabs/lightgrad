# core/matmul.py — matrix multiplication with autograd support

from core.autograd import Tensor

def matmul(a: Tensor, b: Tensor):
    assert len(a.data[0]) == len(b.data), "Incompatible shapes for matmul"

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
            a.grad = matmul(out, transpose(b)).data
        if b.requires_grad:
            b.grad = matmul(transpose(a), out).data

    out._backward = _backward
    out._prev = {a, b}
    return out

def transpose(t: Tensor):
    t_data = list(zip(*t.data))
    return Tensor([list(row) for row in t_data], requires_grad=t.requires_grad)
