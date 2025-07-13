# core/autograd.py

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = [[0.0 for _ in row] for row in data] if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        assert len(self.data) == len(other.data)
        assert len(self.data[0]) == len(other.data[0])

        out_data = [
            [self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))]
            for i in range(len(self.data))
        ]
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                for i in range(len(self.grad)):
                    for j in range(len(self.grad[0])):
                        self.grad[i][j] += out.grad[i][j]
            if other.requires_grad:
                for i in range(len(other.grad)):
                    for j in range(len(other.grad[0])):
                        other.grad[i][j] += out.grad[i][j]
        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        self.grad = [[1.0 for _ in row] for row in self.data]
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

