import random
from core.autograd import Tensor
from core.matmul import matmul, transpose

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights [out_features x in_features]
        self.weight = Tensor([[random.uniform(-0.5, 0.5) for _ in range(in_features)] 
                             for _ in range(out_features)])
        self.weight.requires_grad = True
        
        # Initialize bias as [1 x out_features] for proper broadcasting
        self.bias = Tensor([[random.uniform(-0.5, 0.5) for _ in range(out_features)]])
        self.bias.requires_grad = True
        
        print(f"DEBUG: Linear layer created")
        print(f"  Weight shape: [{len(self.weight.data)}x{len(self.weight.data[0])}]")
        print(f"  Bias shape: [{len(self.bias.data)}x{len(self.bias.data[0])}]")
    
    def __call__(self, x):
        print(f"DEBUG: Linear forward pass")
        print(f"  Input shape: [{len(x.data)}x{len(x.data[0])}]")
        
        # Step 1: Matrix multiplication
        matmul_result = matmul(x, transpose(self.weight))
        print(f"  After matmul: [{len(matmul_result.data)}x{len(matmul_result.data[0])}]")
        
        # Step 2: Add bias
        print(f"  Adding bias: [{len(self.bias.data)}x{len(self.bias.data[0])}]")
        result = matmul_result + self.bias
        print(f"  Final result: [{len(result.data)}x{len(result.data[0])}]")
        
        return result
    
    def parameters(self):
        return [self.weight, self.bias]

