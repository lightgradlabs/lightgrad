from core.autograd import Tensor
from core.loss import mse_loss
from core.matmul import matmul, transpose
import random

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor([[random.uniform(-1, 1) for _ in range(in_features)]
                              for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x: Tensor):
        return matmul(x, transpose(self.weight)) + self.bias

def relu(t: Tensor):
    out_data = [[max(0, val) for val in row] for row in t.data]
    out = Tensor(out_data, requires_grad=t.requires_grad)
    def _backward():
        if t.requires_grad:
            t.grad = [[1 if val > 0 else 0 for val in row] for row in t.data]
    out._backward = _backward
    out._prev = {t}
    return out

# XOR dataset
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
x = Tensor(x_data)
y = Tensor(y_data)

# Model
l1 = Linear(2, 4)
l2 = Linear(4, 1)
lr = 0.1

# Training loop
for epoch in range(50):
    z1 = l1(x)
    a1 = relu(z1)
    z2 = l2(a1)
    pred = z2

    loss = mse_loss(pred, y)
    loss.backward()

    # Manual update
    for i in range(len(l2.weight.data)):
        for j in range(len(l2.weight.data[0])):
            l2.weight.data[i][j] -= lr * (l2.weight.grad[i][j] if l2.weight.grad else 0)
    for j in range(len(l2.bias.data[0])):
        l2.bias.data[0][j] -= lr * (l2.bias.grad[0][j] if l2.bias.grad else 0)

    for i in range(len(l1.weight.data)):
        for j in range(len(l1.weight.data[0])):
            l1.weight.data[i][j] -= lr * (l1.weight.grad[i][j] if l1.weight.grad else 0)
    for j in range(len(l1.bias.data[0])):
        l1.bias.data[0][j] -= lr * (l1.bias.grad[0][j] if l1.bias.grad else 0)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")

