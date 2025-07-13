# core/autograd.py

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data  # 2D list: e.g., [[1.0, 2.0], [3.0, 4.0]]
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set()
        self._backward = lambda: None  # default no-op

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def backward(self):
        self.grad = [[1.0 for _ in row] for row in self.data]  # starting grad = 1.0
        visited = set()
        topo = []

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor([[self.data[i][j] + other for j in range(len(self.data[0]))] 
                          for i in range(len(self.data))])
            return out

        if not isinstance(other, Tensor):
            raise TypeError("Can only add Tensor to Tensor or scalar")

        self_rows, self_cols = len(self.data), len(self.data[0])
        other_rows, other_cols = len(other.data), len(other.data[0])

        if self_rows == other_rows and self_cols == other_cols:
            out_data = [[self.data[i][j] + other.data[i][j] for j in range(self_cols)] 
                        for i in range(self_rows)]

        elif other_rows == 1 and self_cols == other_cols:
            out_data = [[self.data[i][j] + other.data[0][j] for j in range(self_cols)] 
                        for i in range(self_rows)]

        elif self_rows == 1 and self_cols == other_cols:
            out_data = [[self.data[0][j] + other.data[i][j] for j in range(self_cols)] 
                        for i in range(other_rows)]

        else:
            raise ValueError(f"Cannot broadcast shapes [{self_rows}x{self_cols}] and [{other_rows}x{other_cols}]")

        out = Tensor(out_data)
        out.requires_grad = self.requires_grad or other.requires_grad
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = [[0.0 for _ in row] for row in self.data]
                for i in range(len(self.grad)):
                    for j in range(len(self.grad[0])):
                        self.grad[i][j] += out.grad[i][j]

            if other.requires_grad:
                if other.grad is None:
                    other.grad = [[0.0 for _ in row] for row in other.data]
                # Simple version: directly use out.grad (no sum for broadcast reduction)
                if other_rows == 1 and self_cols == other_cols:
                    for j in range(other_cols):
                        grad_sum = sum(out.grad[i][j] for i in range(len(out.grad)))
                        other.grad[0][j] += grad_sum
                else:
                    for i in range(len(other.grad)):
                        for j in range(len(other.grad[0])):
                            other.grad[i][j] += out.grad[i][j]

        out._backward = _backward
        return out
