# https://www.youtube.com/watch?v=OIe48iAqh8E&t=871s
import numpy as np

def low_rank_projection(matrix, rank):
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    # Truncate singular values and reconstruct the low-rank matrix
    U_truncated = U[:, :rank]
    S_truncated = np.diag(S[:rank])
    Vt_truncated = Vt[:rank, :]
    projected_matrix = U_truncated @ S_truncated @ Vt_truncated

    return projected_matrix

# Example usage
np.random.seed(42)
original_matrix = np.random.rand(5, 4)  # 5x4 random matrix
desired_rank = 2
projected_matrix = low_rank_projection(original_matrix, desired_rank)

print("Original matrix:\n", original_matrix)
print("Rank of original matrix:", np.linalg.matrix_rank(original_matrix))
print("Projected matrix (rank 2):\n", projected_matrix)
print("Rank of projected matrix:", np.linalg.matrix_rank(projected_matrix))
print("Original matrix Shape:\n", original_matrix.shape)
print("Projected matrix (rank 2) Shape:\n", projected_matrix.shape)
