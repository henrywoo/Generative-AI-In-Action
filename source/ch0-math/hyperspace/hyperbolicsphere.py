import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_uniform_points_in_poincare_ball(num_points):
    u = np.random.rand(num_points)
    v = np.random.rand(num_points)
    theta = 2 * np.pi * v
    r = np.sqrt(u)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def poincare_to_hyperboloid(x, y):
    d = x ** 2 + y ** 2
    z = np.sqrt(1 + d)
    return x, y, z


def visualize_points_on_hyperboloid(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# 主函数
if __name__ == "__main__":
    num_points = 1000  # 可以根据需要调整点的数量
    x, y = generate_uniform_points_in_poincare_ball(num_points)
    x, y, z = poincare_to_hyperboloid(x, y)
    visualize_points_on_hyperboloid(x, y, z)
