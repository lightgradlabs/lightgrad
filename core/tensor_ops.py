# tensor_ops.py — Reusable math ops for LightGrad

def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def mul(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def matmul(a, b):
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def relu(x):
    return [[max(0, val) for val in row] for row in x]
