# core/layers.py

import random
from core.autograd import Tensor
from core.matmul import matmul, transpose

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)],
            requires_grad=True
        )
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x: Tensor):
        z = matmul(x, transpose(self.weight))
        # Bias broadcasting
        out_data = [
            [z.data[i][j] + self.bias.data[0][j] for j in range(len(z.data[0]))]
            for i in range(len(z.data))
        ]
        out = Tensor(out_data, requires_grad=True)
        out._prev = {x, self.weight, self.bias}
        return out


