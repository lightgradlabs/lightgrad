# LightGrad: Minimal Autograd Engine

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        out = Tensor(
            [[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))]
             for i in range(len(self.data))],
            requires_grad=self.requires_grad or other.requires_grad
        )

        def _backward():
            if self.requires_grad:
                self.grad = [[1 for _ in row] for row in self.data]
            if other.requires_grad:
                other.grad = [[1 for _ in row] for row in other.data]

        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        self._backward()

# Example
if __name__ == "__main__":
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3, 4]], requires_grad=True)
    c = a + b
    print("Forward:", c.data)
    c.backward()
    print("Grad A:", a.grad)
    print("Grad B:", b.grad)
