class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        # Handle scalar addition
        if isinstance(other, (int, float)):
            out = Tensor([[self.data[i][j] + other for j in range(len(self.data[0]))]
                         for i in range(len(self.data))], requires_grad=self.requires_grad)
            return out

        if not isinstance(other, Tensor):
            raise TypeError("Unsupported type for addition")

        self_rows, self_cols = len(self.data), len(self.data[0])
        other_rows, other_cols = len(other.data), len(other.data[0])

        # Case 1: Same shape
        if self_rows == other_rows and self_cols == other_cols:
            out_data = [[self.data[i][j] + other.data[i][j] for j in range(self_cols)]
                        for i in range(self_rows)]

        # Case 2: [m x n] + [1 x n]
        elif other_rows == 1 and self_cols == other_cols:
            out_data = [[self.data[i][j] + other.data[0][j] for j in range(self_cols)]
                        for i in range(self_rows)]

        # Case 3: [1 x n] + [m x n]
        elif self_rows == 1 and self_cols == other_cols:
            out_data = [[self.data[0][j] + other.data[i][j] for j in range(self_cols)]
                        for i in range(other_rows)]

        # Case 4: [m x n] + [m x 1]
        elif other_cols == 1 and self_rows == other_rows:
            out_data = [[self.data[i][j] + other.data[i][0] for j in range(self_cols)]
                        for i in range(self_rows)]

        # Case 5: [m x 1] + [m x n]
        elif self_cols == 1 and self_rows == other_rows:
            out_data = [[self.data[i][0] + other.data[i][j] for j in range(other_cols)]
                        for i in range(self_rows)]
        else:
            raise ValueError("Incompatible shapes for broadcasting")

        out = Tensor(out_data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return

            # Gradients to self
            if self.requires_grad:
                if self.grad is None:
                    self.grad = [[0.0 for _ in range(self_cols)] for _ in range(self_rows)]
                for i in range(self_rows):
                    for j in range(self_cols):
                        self.grad[i][j] += out.grad[i][j]

            # Gradients to other (with reduction)
            if other.requires_grad:
                if other.grad is None:
                    other.grad = [[0.0 for _ in range(other_cols)] for _ in range(other_rows)]

                if other_rows == 1:
                    for j in range(other_cols):
                        grad_sum = sum(out.grad[i][j] for i in range(len(out.grad)))
                        other.grad[0][j] += grad_sum
                elif other_cols == 1:
                    for i in range(other_rows):
                        grad_sum = sum(out.grad[i])
                        other.grad[i][0] += grad_sum
                else:
                    for i in range(other_rows):
                        for j in range(other_cols):
                            other.grad[i][j] += out.grad[i][j]

        out._backward = _backward
        return out

    def backward(self):
        if self.grad is None:
            # Initialize grad as 1.0 for loss scalar
            self.grad = [[1.0]]
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()
