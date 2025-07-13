# LightGrad: Minimal Tensor Ops (CPU-based)

def matmul(a, b):
    # Basic matrix multiplication without NumPy
    result = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

# Example usage (temporary test)
if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    print("Result of matmul:")
    print(matmul(A, B))
