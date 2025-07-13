# XOR training using LightGrad autograd engine (manual gradients)

import random

# Minimal Tensor
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = [[0 for _ in row] for row in data] if requires_grad else None

    def backward(self):
        pass  # Placeholder

# Mean Squared Error Loss
def mse_loss(pred, target):
    diff = [
        [pred.data[i][0] - target.data[i][0]]
        for i in range(len(pred.data))
    ]
    loss_val = sum(d[0]**2 for d in diff) / len(diff)
    grad = [
        [2 * (pred.data[i][0] - target.data[i][0]) / len(diff)]
        for i in range(len(diff))
    ]
    loss = Tensor([[loss_val]], requires_grad=True)
    loss.grad = grad
    return loss

# ReLU Activation
def relu(t: Tensor):
    out_data = [[max(0, val) for val in row] for row in t.data]
    return Tensor(out_data, requires_grad=t.requires_grad)

# Simple Linear Layer
class Linear:
    def __init__(self, in_features, out_features):
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)], requires_grad=True)
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x: Tensor):
        out_data = []
        for i in range(len(x.data)):
            row = []
            for j in range(len(self.weight.data)):
                dot = sum(
                    x.data[i][k] * self.weight.data[j][k]
                    for k in range(len(x.data[0]))
                )
                row.append(dot + self.bias.data[0][j])
            out_data.append(row)
        return Tensor(out_data, requires_grad=True)

# XOR Dataset
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
    out1 = l1(x)
    out2 = relu(out1)
    out3 = l2(out2)
    pred = out3

    loss = mse_loss(pred, y)

    # Manual Gradients (Fake autograd)
    grad = [
        [2 * (pred.data[i][0] - y.data[i][0]) / len(y.data)]
        for i in range(len(y.data))
    ]

    for i in range(len(l2.weight.data)):
        for j in range(len(l2.weight.data[0])):
            l2.weight.data[i][j] -= lr * grad[i % 4][0]
    for j in range(len(l2.bias.data[0])):
        l2.bias.data[0][j] -= lr * grad[0][0]

    for i in range(len(l1.weight.data)):
        for j in range(len(l1.weight.data[0])):
            l1.weight.data[i][j] -= lr * grad[i % 4][0]
    for j in range(len(l1.bias.data[0])):
        l1.bias.data[0][j] -= lr * grad[0][0]

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")
