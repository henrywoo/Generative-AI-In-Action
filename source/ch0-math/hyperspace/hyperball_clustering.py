import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import norm, cdist

class HyperSphere:
    def __init__(self, num_super_points, elev, azim, dim):
        self.num_super_points = num_super_points
        self.elev = elev
        self.azim = azim
        self.dim = dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.script_name = os.path.splitext(os.path.basename(__file__))[0]
        self.imgnum = 0
        self.super_points = self.generate_super_points()

    def generate_super_points(self):
        # Generate points uniformly on a high-dimensional sphere
        indices = np.arange(0, self.num_super_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / self.num_super_points)
        theta = np.pi * (1 + 5 ** 0.5) * indices

        if self.dim == 2:
            x = np.cos(theta)
            y = np.sin(theta)
            points = np.vstack((x, y)).T
        elif self.dim == 3:
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points = np.vstack((x, y, z)).T
        else:
            points = np.random.normal(size=(self.num_super_points, self.dim))
            points /= np.linalg.norm(points, axis=1, keepdims=True)

        return points

    def map_random_points_to_sphere(self, random_points):
        # Ensure the points are torch tensors and move them to the GPU
        random_points = torch.tensor(random_points, dtype=torch.float32).to(self.device)
        self.super_points = torch.tensor(self.super_points, dtype=torch.float32).to(self.device)

        # Normalize the random points to lie on the unit sphere
        norms = norm(random_points, dim=-1, keepdim=True)
        mapped_points = random_points / norms

        # Compute Euclidean distances
        distances = cdist(mapped_points, self.super_points)

        # Find the nearest super points
        indices = torch.argmin(distances, dim=1)
        quantized = self.super_points[indices]

        return mapped_points, quantized

    def save_figure(self, fig):
        filename = f"{self.script_name}_{self.imgnum}.png"
        fig.savefig(filename)
        print(f"Figure saved as {filename}")

    def plot_points_2d(self, random_points, mapped_points, quantized):
        plt.scatter(self.super_points[:, 0], self.super_points[:, 1], c='b', label='Super Points')
        plt.scatter(random_points[:, 0], random_points[:, 1], c='g', alpha=0.5, label='Random Points')
        plt.scatter(mapped_points[:, 0], mapped_points[:, 1], c='c', label='Mapped Points on Unit Sphere')
        plt.scatter(quantized[:, 0], quantized[:, 1], c='r', label='Final Mapped Points')
        plt.title('Mapping Random Points to Nearest Super Points on a Circle')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_points_3d(self, random_points, mapped_points, quantized):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.super_points[:, 0], self.super_points[:, 1], self.super_points[:, 2], c='b', label='Super Points')
        ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], c='g', alpha=0.5, label='Random Points')
        ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='c', label='Mapped Points on Unit Sphere')
        ax.scatter(quantized[:, 0], quantized[:, 1], quantized[:, 2], c='r', label='Final Mapped Points')

        for i in range(random_points.shape[0]):
            ax.plot([random_points[i, 0], quantized[i, 0]],
                    [random_points[i, 1], quantized[i, 1]],
                    [random_points[i, 2], quantized[i, 2]], 'k--')

        ax.view_init(elev=self.elev, azim=self.azim)
        ax.set_title('Mapping Random Points to Nearest Super Points on a Sphere', fontsize=9)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=8)
        plt.grid(True)

        self.save_figure(fig)
        plt.show()

    def run(self, random_points):
        # 将随机点映射到单位球面并找到最近的超级点
        mapped_points, quantized = self.map_random_points_to_sphere(random_points)

        # Move results back to CPU for plotting
        mapped_points = mapped_points.cpu().numpy()
        quantized = quantized.cpu().numpy()

        # 绘制图形
        if self.dim == 2:
            self.plot_points_2d(random_points, mapped_points, quantized)
        elif self.dim == 3:
            self.plot_points_3d(random_points, mapped_points, quantized)
        else:
            print(f"Plotting for {self.dim} dimensions is not supported. Only 2D and 3D are supported for plotting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map random points to nearest super points on a sphere.')
    parser.add_argument('--num_super_points', type=int, default=4096, help='Number of super points (default: 4096)')
    parser.add_argument('--num_random_points', type=int, default=10240, help='Number of random points (default: 10240)')
    parser.add_argument('--elev', type=int, default=30, help='Elevation angle for 3D plot (default: 30)')
    parser.add_argument('--azim', type=int, default=30, help='Azimuth angle for 3D plot (default: 30)')
    parser.add_argument('--dim', type=int, default=3, help='Number of dimensions (default: 3)')

    args = parser.parse_args()

    hypersphere = HyperSphere(args.num_super_points, args.elev, args.azim, args.dim)
    random_points = np.random.uniform(-1, 1, (args.num_random_points, args.dim))
    hypersphere.run(random_points)
