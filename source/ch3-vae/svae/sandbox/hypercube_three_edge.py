import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 将点映射到单位立方体的边上
def map_to_cube_edge(v):
    N = len(v)
    if N < 2:
        raise ValueError("The dimension of the vector must be at least 2")

    abs_v = np.abs(v)
    max_indices = abs_v.argsort()[-2:]  # 找到绝对值最大的两个元素的索引
    u = np.zeros_like(v)
    u[max_indices] = np.sign(v[max_indices])  # 将这两个元素设为 ±1
    return u


# 生成随机3D点
np.random.seed(42)
points = np.random.randn(50, 3)  # 只显示20个点

# 将每个点映射到单位立方体的边上
edge_points = np.array([map_to_cube_edge(point) for point in points])

# 绘制转换前后的点
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原始点
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Original Points')

# 映射到单位立方体边上的点
ax.scatter(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2], c='b', marker='^', label='Edge Points')

# 用虚线连接原始点和映射点
for i in range(points.shape[0]):
    ax.plot([points[i, 0], edge_points[i, 0]],
            [points[i, 1], edge_points[i, 1]],
            [points[i, 2], edge_points[i, 2]], 'k--')

ax.set_title('Original and Edge Points on Cube')
ax.legend()

plt.show()
