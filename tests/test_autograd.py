# Minimal test_autograd.py for LightGrad

from core.autograd import Tensor

def test_add():
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3, 4]], requires_grad=True)
    c = a + b
    assert c.data == [[4, 6]], f"Expected [[4, 6]], got {c.data}"

def test_backward():
    a = Tensor([[1, 2]], requires_grad=True)
    b = Tensor([[3, 4]], requires_grad=True)
    c = a + b
    c.backward()
    assert a.grad == [[1, 1]], f"Expected grad [[1, 1]], got {a.grad}"
    assert b.grad == [[1, 1]], f"Expected grad [[1, 1]], got {b.grad}"

# Run tests manually (since no pytest yet)
test_add()
test_backward()
print("✅ All LightGrad tests passed.")
