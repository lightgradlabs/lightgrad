# LightGrad: Minimal Autograd Engine (Updated with Broadcasting)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(
                [[other for _ in self.data[0]] for _ in self.data],
                requires_grad=False
            )

        # Broadcasting-safe addition
        out_rows = max(len(self.data), len(other.data))
        out_cols = max(len(self.data[0]), len(other.data[0]))

        def get_val(tensor, i, j):
            row = tensor.data[i % len(tensor.data)]
            return row[j % len(row)]

        out_data = [
            [get_val(self, i, j) + get_val(other, i, j) for j in range(out_cols)]
            for i in range(out_rows)
        ]

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
