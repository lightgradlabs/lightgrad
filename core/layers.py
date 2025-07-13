import random
from core.autograd import Tensor
from core.matmul import matmul, transpose

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Tensor([[random.uniform(-0.5, 0.5) for _ in range(in_features)] 
                             for _ in range(out_features)])
        self.weight.requires_grad = True
        
        self.bias = Tensor([[random.uniform(-0.5, 0.5) for _ in range(out_features)]])
        self.bias.requires_grad = True
    
    def __call__(self, x):
        return matmul(x, transpose(self.weight)) + self.bias
    
    def parameters(self):
        return [self.weight, self.bias]

