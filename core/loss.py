# LightGrad Loss Functions

def mse_loss(predicted, target):
    """
    Mean Squared Error Loss: (1/n) * sum((pred - target)^2)
    """
    n = len(predicted) * len(predicted[0])
    total = 0
    for i in range(len(predicted)):
        for j in range(len(predicted[0])):
            diff = predicted[i][j] - target[i][j]
            total += diff * diff
    return total / n

def cross_entropy_loss(predicted, target):
    """
    Simple Cross Entropy Loss for 1D lists (for now)
    predicted = [[0.2, 0.8]]  → probs (softmax)
    target    = [[0,   1]]
    """
    import math
    loss = 0
    for i in range(len(predicted[0])):
        p = predicted[0][i]
        t = target[0][i]
        if p > 0:
            loss -= t * math.log(p)
    return loss
