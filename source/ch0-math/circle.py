import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from hiq import deterministic

# 获取脚本文件名（不包含扩展名）
script_name = os.path.splitext(os.path.basename(__file__))[0]
# 初始化图像编号
imgnum = 0

def save_figure(fig, script_name, imgnum):
    filename = f"{script_name}_{imgnum}.png"
    fig.savefig(filename)
    print(f"Figure saved as {filename}")

# 点的数量
num_super_points = 128
num_random_points = 50

# 生成128个均匀分布在圆上的超级点
angles = np.linspace(0, 2 * np.pi, num_super_points, endpoint=False)
super_points = np.vstack((np.cos(angles), np.sin(angles))).T

# 生成一些随机点
random_points = np.random.uniform(-1, 1, (num_random_points, 2))

# 将随机点映射到单位圆
mapped_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)

# 使用KD树找到最近的超级点
tree = cKDTree(super_points)
_, indices = tree.query(mapped_points)
final_mapped_points = super_points[indices]

# 绘制图形
fig = plt.figure(figsize=(8, 8))
plt.plot(super_points[:, 0], super_points[:, 1], 'bo', label='Super Points')  # 超级点
plt.plot(random_points[:, 0], random_points[:, 1], 'go', label='Random Points', alpha=0.5)  # 随机点
plt.plot(mapped_points[:, 0], mapped_points[:, 1], 'co', label='Mapped Points on Unit Circle')  # 映射到单位圆上的点
plt.plot(final_mapped_points[:, 0], final_mapped_points[:, 1], 'ro', label='Final Mapped Points')  # 最终映射的超级点
for i in range(num_random_points):
    plt.plot([random_points[i, 0], final_mapped_points[i, 0]], [random_points[i, 1], final_mapped_points[i, 1]], 'k--')

# 绘制圆周
theta = np.linspace(0, 2 * np.pi, 1000)
plt.plot(np.cos(theta), np.sin(theta), 'r--')

# 设置图形属性
plt.gca().set_aspect('equal')  # 确保圆形比例
plt.title('Mapping Random Points to Nearest Super Points on a Circle')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
save_figure(fig, script_name, imgnum)
plt.show()
