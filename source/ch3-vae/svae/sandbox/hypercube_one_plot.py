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
    return v_normalized# / np.sqrt(N)

# Generate random 3D points
np.random.seed(42)
points = np.random.randn(50, 3)  # 只显示20个点

# Apply simple_max_norm to each point
normalized_points = np.array([simple_max_norm(point) for point in points])

# Plotting the points before and after normalization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Original Points')

# Normalized points
ax.scatter(normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2], c='b', marker='^', label='Normalized Points')

# Draw lines connecting original and normalized points
for i in range(points.shape[0]):
    ax.plot([points[i, 0], normalized_points[i, 0]],
            [points[i, 1], normalized_points[i, 1]],
            [points[i, 2], normalized_points[i, 2]], 'k--')

ax.set_title('Original and Normalized Points')
ax.legend()

plt.show()
