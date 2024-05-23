import torch
import numpy as np

# Define vectors using PyTorch
A_torch = torch.tensor([1, 2, 3], dtype=torch.float32)
B_torch = torch.tensor([4, 5, 6], dtype=torch.float32)

# Calculate dot product using torch.dot
dot_product_torch_dot = torch.dot(A_torch, B_torch)
print("Dot product using torch.dot:", dot_product_torch_dot.item())

# Calculate dot product using matrix multiplication (@)
A_reshaped = A_torch.view(1, -1)  # Reshape A to 1x3
B_reshaped = B_torch.view(-1, 1)  # Reshape B to 3x1
dot_product_matrix_mul = A_reshaped @ B_reshaped
print("Dot product using matrix multiplication (@):", dot_product_matrix_mul.item())

# Define vectors using NumPy
A_np = np.array([1, 2, 3])
B_np = np.array([4, 5, 6])

# Calculate dot product using np.dot
dot_product_np_dot = np.dot(A_np, B_np)
print("Dot product using np.dot:", dot_product_np_dot)

# Confirm that all methods give the same result
print("\nAll methods give the same result:",
      dot_product_torch_dot.item() == dot_product_matrix_mul.item() == dot_product_np_dot)

