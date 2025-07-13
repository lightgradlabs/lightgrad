from core.autograd import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        import random
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            [[random.uniform(-0.1, 0.1) for _ in range(in_features)] for _ in range(out_features)],
            requires_grad=True
        )
        self.bias = Tensor(
            [[0.0 for _ in range(out_features)]],
            requires_grad=True
        )

    def __call__(self, x: Tensor):
        data = []
        for i in range(len(x.data)):
            row = []
            for j in range(self.out_features):
                dot = sum(
                    x.data[i][k] * self.weight.data[j][k]
                    for k in range(self.in_features)
                )
                row.append(dot + self.bias.data[0][j])
            data.append(row)

        out = Tensor(data, requires_grad=True)

        def _backward():
            # Gradients (placeholder)
            if out.requires_grad:
                out.grad = [[1 for _ in row] for row in out.data]
        out._backward = _backward
        out._prev = {x, self.weight, self.bias}
        return out

