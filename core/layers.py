# core/layers.py — using autograd + matmul engine

import random
from core.autograd import Tensor
from core.matmul import matmul

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)],
            requires_grad=True
        )
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x: Tensor):
        return matmul(x, transpose(self.weight)) + self.bias

def transpose(t: Tensor):
    t_data = list(zip(*t.data))
    return Tensor([list(row) for row in t_data], requires_grad=t.requires_grad)

