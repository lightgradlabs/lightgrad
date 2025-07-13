# XOR training with improved manual gradient updates

import random

# Tensor Class
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
    return Tensor([[loss_val]]), grad

# ReLU Activation
def relu(t: Tensor):
    out_data = [[max(0, val) for val in row] for row in t.data]
    return Tensor(out_data, requires_grad=t.requires_grad)

# Linear Layer
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
        self.last_input = x.data  # Save input for manual backprop
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

    def manual_update(self, grad_output, lr):
        for i in range(self.out_features):
            for j in range(self.in_features):
                grad_sum = sum(
                    grad_output[b][i] * self.last_input[b][j]
                    for b in range(len(self.last_input))
                ) / len(self.last_input)
                self.weight.data[i][j] -= lr * grad_sum

        for j in range(self.out_features):
            grad_sum = sum(grad_output[b][j] for b in range(len(self.last_input))) / len(self.last_input)
            self.bias.data[0][j] -= lr * grad_sum

# XOR Data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
x = Tensor(x_data)
y = Tensor(y_data)

# Model
l1 = Linear(2, 4)
l2 = Linear(4, 1)
lr = 0.1

# Training
for epoch in range(50):
    out1 = l1(x)
    out2 = relu(out1)
    out3 = l2(out2)
    pred = out3

    loss, loss_grad = mse_loss(pred, y)

    # Update weights with fake gradients
    l2.manual_update(loss_grad, lr)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")
