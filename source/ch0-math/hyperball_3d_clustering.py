import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from hiq import deterministic


def generate_super_points(num_super_points):
    indices = np.arange(0, num_super_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_super_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T


def map_random_points_to_sphere(random_points, super_points):
    norms = np.linalg.norm(random_points, axis=1, keepdims=True)
    mapped_points = random_points / norms
    tree = cKDTree(super_points)
    _, indices = tree.query(mapped_points)
    final_mapped_points = super_points[indices]
    return mapped_points, final_mapped_points


def save_figure(fig, script_name, imgnum):
    filename = f"{script_name}_{imgnum}.png"
    fig.savefig(filename)
    print(f"Figure saved as {filename}")


def plot_points(super_points, random_points, mapped_points, final_mapped_points, elev, azim, script_name, imgnum):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(super_points[:, 0], super_points[:, 1], super_points[:, 2], c='b', label='Super Points')
    ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], c='g', alpha=0.5, label='Random Points')
    ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='c',
               label='Mapped Points on Unit Sphere')
    ax.scatter(final_mapped_points[:, 0], final_mapped_points[:, 1], final_mapped_points[:, 2], c='r',
               label='Final Mapped Points')

    for i in range(random_points.shape[0]):
        ax.plot([random_points[i, 0], final_mapped_points[i, 0]],
                [random_points[i, 1], final_mapped_points[i, 1]],
                [random_points[i, 2], final_mapped_points[i, 2]], 'k--')

    ax.view_init(elev=elev, azim=azim)
    ax.set_title('Mapping Random Points to Nearest Super Points on a Sphere', fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(fontsize=8)
    plt.grid(True)

    save_figure(fig, script_name, imgnum)
    plt.show()


def main(num_super_points, num_random_points, elev, azim):
    # 获取脚本文件名（不包含扩展名）
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    # 初始化图像编号
    imgnum = 0

    # 生成超级点
    super_points = generate_super_points(num_super_points)

    # 生成一些随机点
    random_points = np.random.uniform(-1, 1, (num_random_points, 3))

    # 将随机点映射到单位球面并找到最近的超级点
    mapped_points, final_mapped_points = map_random_points_to_sphere(random_points, super_points)

    # 绘制图形
    plot_points(super_points, random_points, mapped_points, final_mapped_points, elev, azim, script_name, imgnum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map random points to nearest super points on a sphere.')
    parser.add_argument('--num_super_points', type=int, default=256, help='Number of super points (default: 256)')
    parser.add_argument('--num_random_points', type=int, default=50, help='Number of random points (default: 50)')
    parser.add_argument('--elev', type=int, default=30, help='Elevation angle for 3D plot (default: 30)')
    parser.add_argument('--azim', type=int, default=30, help='Azimuth angle for 3D plot (default: 30)')

    args = parser.parse_args()

    main(args.num_super_points, args.num_random_points, args.elev, args.azim)
