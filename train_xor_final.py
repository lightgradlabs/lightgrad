# XOR training using LightGrad full stack (autograd + matmul)

from core.autograd import Tensor
from core.loss import mse_loss
from core.layers import Linear
from core.tensor_ops import add  # for bias
import random

# XOR data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]

x = Tensor(x_data, requires_grad=False)
y = Tensor(y_data, requires_grad=False)

# Model
l1 = Linear(2, 4)
l2 = Linear(4, 1)
lr = 0.1

def relu(t: Tensor):
    relu_data = [[max(0, val) for val in row] for row in t.data]
    out = Tensor(relu_data, requires_grad=t.requires_grad)
    out._prev = {t}

    def _backward():
        if t.requires_grad:
            t.grad = [[1 if val > 0 else 0 for val in row] for row in t.data]

    out._backward = _backward
    return out

# Training loop
for epoch in range(50):
    z1 = l1(x)
    a1 = relu(z1)
    z2 = l2(a1)
    pred = z2

    loss = mse_loss(pred, y)
    loss.backward()

    # Gradient descent
    for i in range(len(l2.weight.data)):
        for j in range(len(l2.weight.data[0])):
            l2.weight.data[i][j] -= lr * l2.weight.grad[i][j]
    for j in range(len(l2.bias.data[0])):
        l2.bias.data[0][j] -= lr * l2.bias.grad[0][j]

    for i in range(len(l1.weight.data)):
        for j in range(len(l1.weight.data[0])):
            l1.weight.data[i][j] -= lr * l1.weight.grad[i][j]
    for j in range(len(l1.bias.data[0])):
        l1.bias.data[0][j] -= lr * l1.bias.grad[0][j]

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")
