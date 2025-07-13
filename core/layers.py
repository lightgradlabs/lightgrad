# LightGrad Linear Layer

from core.autograd import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        import random
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and bias with small random values
        self.weight = Tensor([[random.uniform(-0.1, 0.1) for _ in range(in_features)] for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([[0.0] * out_features], requires_grad=True)

    def __call__(self, x):
        # Forward pass: x @ W^T + b
        out = []
        for i in range(len(x)):
            row = []
            for j in range(self.out_features):
                dot = sum(x[i][k] * self.weight.data[j][k] for k in range(self.in_features))
                row.append(dot + self.bias.data[0][j])
            out.append(row)
        return out
