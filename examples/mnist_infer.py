# LightGrad MNIST Inference Demo (Simulated)

from core.tensor_ops import matmul

# Simulated input and weight matrix
input_data = [[0.1, 0.2], [0.3, 0.4]]
weights = [[0.5, 0.6], [0.7, 0.8]]

output = matmul(input_data, weights)

print("🧠 LightGrad Output:")
for row in output:
    print(row)

