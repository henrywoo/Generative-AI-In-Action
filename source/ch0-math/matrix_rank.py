import numpy as np

# 创建全零矩阵
zero_matrix = np.zeros((4, 4))

# 计算全零矩阵的秩
rank_zero_matrix = np.linalg.matrix_rank(zero_matrix)
print("Rank of zero matrix:", rank_zero_matrix)

# 创建元素为e^0的矩阵
exp_matrix = np.exp(zero_matrix)

# 计算元素为e^0的矩阵的秩
rank_exp_matrix = np.linalg.matrix_rank(exp_matrix)
print("Rank of exp(zero matrix):", rank_exp_matrix)

n = 100
x = np.random.randn(n)
y = x[:, None].dot(x[None])

print(np.linalg.matrix_rank(y)) # 秩为1
print(np.linalg.matrix_rank(np.exp(y))) # 秩大概率为17、18

# 创建原始矩阵
original_matrix = np.array([[0, 1, 2],
                            [3, 4, 5],
                            [3, 4, 5]])

# 计算原始矩阵的秩
rank_original_matrix = np.linalg.matrix_rank(original_matrix)
print("Rank of original matrix:", rank_original_matrix)

# 对每个元素取指数
exp_matrix = np.exp(original_matrix)

# 计算取指数后矩阵的秩
rank_exp_matrix = np.linalg.matrix_rank(exp_matrix)
print("Rank of exp(original matrix):", rank_exp_matrix)
