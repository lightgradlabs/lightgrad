# core/autograd.py

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = [[0.0 for _ in row] for row in data] if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor([[other]])
        out_data = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(self.data[0])):
                # Support broadcasting for bias shapes
                if len(other.data) == 1:
                    if len(other.data[0]) == 1:
                        val = other.data[0][0]  # scalar bias
                    else:
                        val = other.data[0][j]  # row vector bias
                else:
                    val = other.data[i][j]
                row.append(self.data[i][j] + val)
            out_data.append(row)

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

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
        return out

    def backward(self):
        if self.grad is None:
            self.grad = [[1.0 for _ in row] for row in self.data]

        visited = set()
        topo = []

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        for t in reversed(topo):
            t._backward()


