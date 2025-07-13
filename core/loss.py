# core/loss.py — MSE Loss for LightGrad

from core.autograd import Tensor

def mse_loss(predicted: Tensor, target: Tensor):
    # Compute difference: predicted - target
    diff = [[predicted.data[i][0] - target.data[i][0]] for i in range(len(predicted.data))]
    
    # Compute squared error
    squared = [[d[0] ** 2] for d in diff]
    
    # Compute mean
    loss_val = sum([s[0] for s in squared]) / len(squared)

    # Return as Tensor object (with gradient tracking enabled)
    loss = Tensor([[loss_val]], requires_grad=True)

    return loss

