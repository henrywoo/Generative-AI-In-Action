import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D

# 获取脚本文件名（不包含扩展名）
script_name = os.path.splitext(os.path.basename(__file__))[0]
# 初始化图像编号
imgnum = 0

def save_figure(fig, script_name, imgnum):
    filename = f"{script_name}_{imgnum}.png"
    fig.savefig(filename)
    print(f"Figure saved as {filename}")

# 点的数量
num_super_points = 32*8
num_random_points = 50

# 生成32个均匀分布在球面上的超级点
indices = np.arange(0, num_super_points, dtype=float) + 0.5
phi = np.arccos(1 - 2*indices/num_super_points)
theta = np.pi * (1 + 5**0.5) * indices

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
super_points = np.vstack((x, y, z)).T

# 生成一些随机点
random_points = np.random.uniform(-1, 1, (num_random_points, 3))

# 将随机点映射到单位球面
norms = np.linalg.norm(random_points, axis=1, keepdims=True)
mapped_points = random_points / norms

# 使用KD树找到最近的超级点
tree = cKDTree(super_points)
_, indices = tree.query(mapped_points)
final_mapped_points = super_points[indices]

# 绘制图形
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(super_points[:, 0], super_points[:, 1], super_points[:, 2], c='b', label='Super Points')  # 超级点
ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], c='g', alpha=0.5, label='Random Points')  # 随机点
ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='c', label='Mapped Points on Unit Sphere')  # 映射到单位球上的点
ax.scatter(final_mapped_points[:, 0], final_mapped_points[:, 1], final_mapped_points[:, 2], c='r', label='Final Mapped Points')  # 最终映射的超级点

for i in range(num_random_points):
    ax.plot([random_points[i, 0], final_mapped_points[i, 0]], [random_points[i, 1], final_mapped_points[i, 1]], [random_points[i, 2], final_mapped_points[i, 2]], 'k--')

# 设置视角
ax.view_init(elev=30, azim=30)  # 这里可以修改视角的 elev 和 azim 参数

# 设置图形属性
ax.set_title('Mapping Random Points to Nearest Super Points on a Sphere', fontsize=9)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.grid(True)

save_figure(fig, script_name, imgnum)
plt.show()
