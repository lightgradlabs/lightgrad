# LightGrad: True Autograd Engine (v2)

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
        visited = set()
        topo_order = []

        def build_graph(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_graph(child)
                topo_order.append(t)

        build_graph(self)

        self.grad = [[1 for _ in row] for row in self.data]  # dL/dL = 1
        for node in reversed(topo_order):
            node._backward()


# Example
if __name__ == "__main__":
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3, 4]], requires_grad=True)
    c = a + b
    print("Forward:", c.data)
    c.backward()
    print("Grad A:", a.grad)
    print("Grad B:", b.grad)
