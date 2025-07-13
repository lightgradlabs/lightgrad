# train.py — Train simple linear model with LightGrad

from core.autograd import Tensor
from core.tensor_ops import add, mul
from core.loss import mse_loss

# Inputs and target (X → y)
x_data = [[1.0, 2.0]]  # shape: [1 x 2]
y_data = [[10.0]]      # target output

# Initialize weights and bias as Tensor
w = Tensor([[0.1, 0.1]], requires_grad=True)  # weights
b = Tensor([[0.0]], requires_grad=True)      # bias

# Learning rate
lr = 0.01

# Training loop
for epoch in range(10):
    # Forward: y_pred = x @ w^T + b
    wx = [[sum(x_data[0][k] * w.data[0][k] for k in range(2))]]
    y_pred = [[wx[0][0] + b.data[0][0]]]

    # Loss
    loss = mse_loss(y_pred, y_data)

    # Manually compute gradients (backprop mockup)
    grad_y_pred = 2 * (y_pred[0][0] - y_data[0][0]) / 1  # batch size = 1
    w.grad = [[grad_y_pred * x_data[0][i] for i in range(2)]]
    b.grad = [[grad_y_pred]]

    # Gradient descent update
    for i in range(2):
        w.data[0][i] -= lr * w.grad[0][i]
    b.data[0][0] -= lr * b.grad[0][0]

    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, w = {w.data}, b = {b.data}")
