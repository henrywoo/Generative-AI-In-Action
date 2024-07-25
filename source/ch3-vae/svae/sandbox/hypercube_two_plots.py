import numpy as np

# 这种方法通过简单的最大值归一化并按比例缩放，实现了一种非常简化的归一化方式。
# 虽然它不严格地将向量放到单位球面上，但计算量非常小，且能在一定程度上保留向量的方向信息。

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the simple_max_norm function
def simple_max_norm(v):
    N = len(v)
    max_val = np.max(np.abs(v))
    if max_val == 0:
        return v  # Avoid division by zero
    v_normalized = v / max_val
    return v_normalized / np.sqrt(N)

# Generate random 3D points
np.random.seed(42)
points = np.random.randn(100, 3)

# Apply simple_max_norm to each point
normalized_points = np.array([simple_max_norm(point) for point in points])

# Plotting the points before and after normalization
fig = plt.figure(figsize=(14, 6))

# Original points
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
ax1.set_title('Original Points')

# Normalized points
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2], c='b', marker='^')
ax2.set_title('Normalized Points using simple_max_norm')

plt.show()
