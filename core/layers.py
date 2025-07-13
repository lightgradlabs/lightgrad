# core/layers.py

import random
from core.autograd import Tensor
from core.matmul import matmul, transpose

class Linear:
    def __init__(self, in_features, out_features):
        import random
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x):
        out = matmul(x, transpose(self.weight))  # Shape: [batch_size x out_features]

        # ✅ Broadcast bias manually
        bias_expanded = Tensor([self.bias.data[0] for _ in range(len(out.data))], requires_grad=True)

        return out + bias_expanded


