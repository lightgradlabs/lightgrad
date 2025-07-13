# core/matmul.py

from core.autograd import Tensor

def matmul(a: Tensor, b: Tensor) -> Tensor:
    out_data = [
        [
            sum(a.data[i][k] * b.data[k][j] for k in range(len(b)))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]
    return Tensor(out_data, requires_grad=a.requires_grad or b.requires_grad)

def transpose(t: Tensor) -> list:
    return list(map(list, zip(*t.data)))

