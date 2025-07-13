# core/loss.py

from core.autograd import Tensor

def mse_loss(predicted, target):
    diff = [
        [predicted.data[i][0] - target.data[i][0]]
        for i in range(len(predicted.data))
    ]
    loss_val = sum([d[0]**2 for d in diff]) / len(diff)
    loss = Tensor([[loss_val]], requires_grad=True)
    loss._backward = lambda: None  # simplified
    return loss
