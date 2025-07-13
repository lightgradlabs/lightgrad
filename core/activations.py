# ReLU Activation for LightGrad

from core.autograd import Tensor

def relu(tensor):
    data = [
        [max(0, value) for value in row]
        for row in tensor.data
    ]
    out = Tensor(data, requires_grad=tensor.requires_grad)

    def _backward():
        if tensor.requires_grad:
            tensor.grad = [
                [1 if val > 0 else 0 for val in row]
                for row in tensor.data
            ]
    out._backward = _backward
    out._prev = {tensor}
    return out
