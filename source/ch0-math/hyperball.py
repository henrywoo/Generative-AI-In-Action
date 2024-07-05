import os
import numpy as np
import matplotlib.pyplot as plt
from hiq import deterministic

# 使用 ggplot 样式
plt.style.use('ggplot')

# 获取脚本文件名（不包含扩展名）
script_name = os.path.splitext(os.path.basename(__file__))[0]

# 初始化图像编号
imgnum = 0

def save_figure(fig, script_name, imgnum):
    filename = f"{script_name}_{imgnum}.png"
    fig.savefig(filename)
    print(f"Figure saved as {filename}")

def demo_2d(imgnum=0):
    # 生成随机点
    num_points = 20
    points = np.random.randn(num_points, 2)  # 生成随机点 (x, y)

    # 将点映射到单位圆上
    mapped_points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # 定义颜色
    colors = plt.cm.Reds(np.linspace(0, 1, num_points))

    # 可视化映射前后的点
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # 映射前的点
    for i in range(num_points):
        ax[0].scatter(points[i, 0], points[i, 1], color=colors[i], label=f'Point {i+1}' if i == 0 else "")
    ax[0].set_title('Original Points')
    ax[0].set_xlim(-3, 3)
    ax[0].set_ylim(-3, 3)
    ax[0].axhline(0, color='grey', lw=0.5)
    ax[0].axvline(0, color='grey', lw=0.5)
    ax[0].legend()
    ax[0].grid(True)

    # 映射后的点（单位圆上的点）
    for i in range(num_points):
        ax[1].scatter(mapped_points[i, 0], mapped_points[i, 1], color=colors[i], label=f'Point {i+1}' if i == 0 else "")
    circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--')
    ax[1].add_artist(circle)
    ax[1].set_title('Mapped Points (Unit Circle)')
    ax[1].set_xlim(-1.5, 1.5)
    ax[1].set_ylim(-1.5, 1.5)
    ax[1].axhline(0, color='grey', lw=0.5)
    ax[1].axvline(0, color='grey', lw=0.5)
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()

    # 保存图像
    save_figure(fig, script_name, imgnum)
    imgnum += 1
    plt.show()


def demo_3d(imgnum=1):
    # 生成随机点
    num_points = 20
    points = np.random.randn(num_points, 3)  # 生成随机点 (x, y, z)

    # 将点映射到单位球面上
    mapped_points = points / np.linalg.norm(points, axis=1, keepdims=True)

    # 定义颜色
    colors = plt.cm.Reds(np.linspace(0, 1, num_points))

    # 创建单位球网格
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # 可视化映射前后的点
    fig = plt.figure(figsize=(10, 5))

    # 映射前的点
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors)
    ax1.set_title('Original Points', fontsize=10)
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([-3, 3])
    ax1.grid(True)

    # 映射后的点（单位球面上的点）
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], color=colors)
    ax2.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)  # 绘制单位球
    ax2.set_title('Mapped Points (Unit Sphere)', fontsize=10)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.grid(True)

    # 保存图像
    #plt.tight_layout()
    save_figure(fig, script_name, imgnum)
    imgnum += 1

    # 显示交互式图像
    plt.show()

def demo_4d(imgnum=2):
    # 生成随机点
    num_points = 20
    points = np.random.randn(num_points, 4)  # 生成随机点 (x, y, z, w)

    # 将点映射到单位4D超球面上
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    mapped_points = points / norms

    # 投影到3D空间（选择前三个坐标）
    projected_points_3d_123 = mapped_points[:, :3]

    # 投影到3D空间（选择后三个坐标）
    projected_points_3d_234 = mapped_points[:, 1:]

    # 投影到2D空间（选择中间两个坐标）
    projected_points_2d_23 = mapped_points[:, 1:3]

    # 定义颜色
    colors = plt.cm.Reds(np.linspace(0, 1, num_points))

    # 创建单位球网格
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # 可视化映射后的点
    fig = plt.figure(figsize=(15, 5))

    # 投影到3D空间（选择前三个坐标）
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(projected_points_3d_123[:, 0], projected_points_3d_123[:, 1], projected_points_3d_123[:, 2],
                color=colors)
    ax1.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)  # 绘制单位球
    ax1.set_title('Projection to 3D (x, y, z)', fontsize=10)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    ax1.grid(True)

    # 投影到3D空间（选择后三个坐标）
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(projected_points_3d_234[:, 0], projected_points_3d_234[:, 1], projected_points_3d_234[:, 2],
                color=colors)
    ax2.plot_surface(x, y, z, color='b', alpha=0.1, rstride=5, cstride=5)  # 绘制单位球
    ax2.set_title('Projection to 3D (y, z, w)', fontsize=10)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    ax2.grid(True)

    # 投影到2D空间（选择中间两个坐标）
    ax3 = fig.add_subplot(133)
    ax3.scatter(projected_points_2d_23[:, 0], projected_points_2d_23[:, 1], color=colors)
    circle = plt.Circle((0, 0), 1, color='b', alpha=0.1, linestyle='--')
    ax3.add_artist(circle)
    ax3.set_title('Projection to 2D (y, z)', fontsize=10)
    ax3.set_xlim([-1.5, 1.5])
    ax3.set_ylim([-1.5, 1.5])
    ax3.axhline(0, color='grey', lw=0.5)
    ax3.axvline(0, color='grey', lw=0.5)
    ax3.grid(True)

    # 保存图像
    plt.tight_layout()
    save_figure(fig, script_name, imgnum)
    imgnum += 1

    # 显示交互式图像
    plt.show()


if __name__ == '__main__':
    demo_2d()
    demo_3d()
    demo_4d()
