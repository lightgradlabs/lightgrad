# /content/lightgrad/core/matmul.py

from core.autograd import Tensor

def matmul(a: Tensor, b: Tensor) -> Tensor:
    assert len(a.data[0]) == len(b.data), "Matrix dimensions do not match for multiplication"

    # DEBUG
    print("DEBUG: matmul called")
    print(f"  a.shape: [{len(a.data)}x{len(a.data[0])}]")
    print(f"  b.shape: [{len(b.data)}x{len(b.data[0])}]")

    result = []
    for i in range(len(a.data)):
        row = []
        for j in range(len(b.data[0])):
            val = sum(a.data[i][k] * b.data[k][j] for k in range(len(b.data)))
            row.append(val)
        result.append(row)

    return Tensor(result)

def transpose(t: Tensor) -> Tensor:
    rows, cols = len(t.data), len(t.data[0])
    transposed = [[t.data[i][j] for i in range(rows)] for j in range(cols)]
    return Tensor(transposed)
