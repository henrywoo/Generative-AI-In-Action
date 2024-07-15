import numpy as np
import matplotlib.pyplot as plt

# 用于缓存计算结果的全局变量
cached_x = None
cached_y = None
cached_params = None


def sample_from_vMF(size, kappa, dims, epsilon=1e-7):
    global cached_x, cached_y, cached_params

    # 检查缓存是否可用
    if cached_params != (kappa, dims, epsilon):
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x ** 2) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]

        # 缓存结果
        cached_x = x
        cached_y = y
        cached_params = (kappa, dims, epsilon)

    # 使用缓存的 y 和 x 进行插值采样
    w = np.interp(np.random.random(size), cached_y, cached_x)

    # 生成与均值方向正交的均匀分布向量
    v = np.random.normal(size=(size, dims - 1))
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # 组合 w 和 v 形成最终的采样结果
    result = np.zeros((size, dims))
    result[:, :-1] = np.sqrt(1 - w ** 2)[:, np.newaxis] * v
    result[:, -1] = w

    return result


def rotate_samples(samples, mu):
    dims = len(mu)
    if np.allclose(mu, np.array([0, 0, 1])):
        return samples

    # 创建旋转矩阵
    mu = mu / np.linalg.norm(mu)
    z = np.array([0, 0, 1])
    v = np.cross(z, mu)
    s = np.linalg.norm(v)
    c = np.dot(z, mu)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(dims) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    return np.dot(samples, R.T)


# 三维情况下的 vMF 分布参数
mu = np.array([1, 0, 0])
kappa = 5
size = 1000

# 采样
samples = sample_from_vMF(size, kappa, dims=3)
samples = rotate_samples(samples, mu)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D von Mises-Fisher Distribution Samples')
plt.show()
