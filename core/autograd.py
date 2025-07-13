# core/autograd.py — LightGrad Autograd Engine

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        # Broadcasting-safe add
        if isinstance(other, Tensor):
            out_data = [
                [self.data[i][j] + other.data[i % len(other.data)][j % len(other.data[0])]
                 for j in range(len(self.data[0]))]
                for i in range(len(self.data))
            ]
            requires_grad = self.requires_grad or other.requires_grad
        else:
            # Scalar or list
            out_data = [[self.data[i][j] + other for j in range(len(self.data[0]))]
                        for i in range(len(self.data))]
            requires_grad = self.requires_grad

        out = Tensor(out_data, requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = [[1 for _ in row] for row in self.data]
            if isinstance(other, Tensor) and other.requires_grad:
                other.grad = [[1 for _ in row] for row in other.data]

        out._backward = _backward
        out._prev = {self} | ({other} if isinstance(other, Tensor) else set())
        return out

    def __matmul__(self, other):
        out_data = [
            [sum(self.data[i][k] * other.data[k][j] for k in range(len(other.data)))
             for j in range(len(other.data[0]))]
            for i in range(len(self.data))
        ]
        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                # Gradient w.r.t self = grad_output @ other.T
                pass  # Placeholder
            if other.requires_grad:
                # Gradient w.r.t other = self.T @ grad_output
                pass  # Placeholder

        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        topo = []
        visited = set()

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

# Optional test
if __name__ == "__main__":
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3, 4]], requires_grad=True)
    c = a + b
    print("Forward:", c.data)
    c.backward()
    print("Grad A:", a.grad)
    print("Grad B:", b.grad)

