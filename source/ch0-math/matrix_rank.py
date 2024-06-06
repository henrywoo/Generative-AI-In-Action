import numpy as np

# Create a zero matrix
zero_matrix = np.zeros((4, 4))
print("Zero matrix:")
print(zero_matrix)

# Calculate the rank of the zero matrix
rank_zero_matrix = np.linalg.matrix_rank(zero_matrix)
print("Rank of zero matrix:", rank_zero_matrix)

# Create a matrix with elements e^0
exp_matrix_zero = np.exp(zero_matrix)
print("Matrix with elements e^0 (exp(zero matrix)):")
print(exp_matrix_zero)

# Calculate the rank of the matrix with elements e^0
rank_exp_matrix_zero = np.linalg.matrix_rank(exp_matrix_zero)
print("Rank of exp(zero matrix):", rank_exp_matrix_zero)

n = 5
x = np.random.randn(n)
y = x[:, None] @ x[None]
print("Matrix y (outer product of random vector x):")
print(y)

# Calculate the rank of y
rank_y = np.linalg.matrix_rank(y)
print("Rank of matrix y:", rank_y)

# Apply exponential function to matrix y
exp_y = np.exp(y)
print("Matrix exp(y) after exponential:")
print(exp_y)

# Calculate the rank of exp(y)
rank_exp_y = np.linalg.matrix_rank(exp_y)
print("Rank of exp(y):", rank_exp_y)

# Create the original matrix
original_matrix = np.array([[0, 1, 2],
                            [3, 4, 5],
                            [3, 4, 5]])
print("Original matrix:")
print(original_matrix)

# Calculate the rank of the original matrix
rank_original_matrix = np.linalg.matrix_rank(original_matrix)
print("Rank of original matrix:", rank_original_matrix)

# Apply the exponential function to each element of the original matrix
exp_matrix_original = np.exp(original_matrix)
print("Matrix with elements exp(original matrix):")
print(exp_matrix_original)

# Calculate the rank of the matrix after applying the exponential function
rank_exp_matrix_original = np.linalg.matrix_rank(exp_matrix_original)
print("Rank of exp(original matrix):", rank_exp_matrix_original)
