# train_mlp.py — Train LightGrad MLP on XOR

from core.autograd import Tensor
from core.loss import mse_loss
from core.layers import Linear
from core.activations import relu

# XOR dataset
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y = [
    [0],
    [1],
    [1],
    [0]
]

# Wrap input/output in Tensors
x = Tensor(X, requires_grad=False)
y = Tensor(Y, requires_grad=False)

# Model: Linear → ReLU → Linear
l1 = Linear(2, 4)
l2 = Linear(4, 1)

lr = 0.1  # learning rate

for epoch in range(50):
    # Forward
    out1 = l1(x.data)
    out1_tensor = Tensor(out1, requires_grad=True)

    out2 = relu(out1_tensor)
    out3 = l2(out2.data)
    out3_tensor = Tensor(out3, requires_grad=True)

    # Loss
    pred = out3_tensor
    loss = mse_loss(pred, y)
    
    # Backward
    loss.backward()

    # Manual gradient descent
    for param in [l1.weight, l1.bias, l2.weight, l2.bias]:
        for i in range(len(param.data)):
            for j in range(len(param.data[0])):
                param.data[i][j] -= lr * (param.grad[i][j] if param.grad else 0)

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Loss = {round(loss.data[0][0], 4)}")
