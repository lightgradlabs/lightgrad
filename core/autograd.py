# LightGrad: Minimal Autograd Engine with Broadcasting Add

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def shape(self):
        return len(self.data), len(self.data[0]) if self.data else 0

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Only Tensor addition supported.")

        a_rows, a_cols = self.shape()
        b_rows, b_cols = other.shape()

        out_data = []
        for i in range(max(a_rows, b_rows)):
            row = []
            for j in range(max(a_cols, b_cols)):
                a_val = self.data[i % a_rows][j % a_cols]
                b_val = other.data[i % b_rows][j % b_cols]
                row.append(a_val + b_val)
            out_data.append(row)

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

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
