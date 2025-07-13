# XOR training using LightGrad autograd engine

from core.autograd import Tensor
import random

# Manual Linear Layer
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            [[random.uniform(-1, 1) for _ in range(in_features)]
             for _ in range(out_features)],
            requires_grad=True
        )
        self.bias = Tensor([[0.0 for _ in range(out_features)]], requires_grad=True)

    def __call__(self, x: Tensor):
        out_data = []
        for i in range(len(x.data)):
            row = []
            for j in range(self.out_features):
                dot = sum(
                    x.data[i][k] * self.weight.data[j][k]
                    for k in range(self.in_features)
                )
                row.append(dot + self.bias.data[0][j])
            out_data.append(row)
        return Tensor(out_data, requires_grad=True)

# ReLU Activation
def relu(t: Tensor):
    return Tensor([[max(0, val) for val in row] for row in t.data], requires_grad=True)

# MSE Loss
def mse_loss(pred: Tensor, target: Tensor):
    loss_val = sum(
        (pred.data[i][0] - target.data[i][0]) ** 2
        for i in range(len(pred.data))
    ) / len(pred.data)
    return Tensor([[loss_val]], requires_grad=True)

# XOR Data
x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = Tensor([[0], [1], [1], [0]])

# Model
l1 = Linear(2, 4)
l2 = Linear(4, 1)
lr = 0.1

# Training loop
for epoch in range(50):
    out1 = l1(x)
    out2 = relu(out1)
    pred = l2(out2)

    loss = mse_loss(pred, y)
    loss.backward()  # Real autograd backprop

    # Manual parameter updates using gradients
    for i in range(l2.weight.out_features):
        for j in range(l2.weight.in_features):
            l2.weight.data[i][j] -= lr * l2.weight.grad[i][j]
    for j in range(l2.bias.out_features):
        l2.bias.data[0][j] -= lr * l2.bias.grad[0][j]

    for i in range(l1.weight.out_features):
        for j in range(l1.weight.in_features):
            l1.weight.data[i][j] -= lr * l1.weight.grad[i][j]
    for j in range(l1.bias.out_features):
        l1.bias.data[0][j] -= lr * l1.bias.grad[0][j]

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")
