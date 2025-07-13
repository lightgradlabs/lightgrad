# core/autograd.py — LightGrad autograd engine (final fixed)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            [[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))]
             for i in range(len(self.data))],
            requires_grad=self.requires_grad or other.requires_grad
        )

        def _backward():
            if self.requires_grad:
                self.grad = [[1.0 for _ in row] for row in self.data]
            if other.requires_grad:
                other.grad = [[1.0 for _ in row] for row in other.data]

        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        # Topological sort (simple)
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        self.grad = [[1.0 for _ in row] for row in self.data]

        for t in reversed(topo):
            t._backward()
