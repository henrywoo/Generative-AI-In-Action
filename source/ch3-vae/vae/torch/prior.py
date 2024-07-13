import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 生成1000个三维正态分布的样本点
num_points = 2048
mean = np.zeros(3)  # 均值为0
cov = np.eye(3)     # 协方差矩阵为单位矩阵

# 生成多元正态分布的样本
points = np.random.multivariate_normal(mean, cov, num_points)

# 创建一个三维图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', alpha=0.6)

# 设置图形的标题和轴标签
ax.set_title('3D Visualization of Multivariate Normal Distribution')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()
