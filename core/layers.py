# core/layers.py

import random
from core.autograd import Tensor
from core.matmul import matmul, transpose

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x):
        wx = matmul(x, transpose(self.weight))
        # Broadcast bias across all rows
        bias_expanded = Tensor([self.bias.data[0] for _ in range(len(wx.data))], requires_grad=True)
        return wx + bias_expanded

