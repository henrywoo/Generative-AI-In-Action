import numpy as np
import matplotlib.pyplot as plt


def sample_vMF(mu, kappa, size=1):
    dim = len(mu)
    result = np.zeros((size, dim))

    b = (-2 * kappa + (4 * kappa ** 2 + (dim - 1) ** 2) ** 0.5) / (dim - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (dim - 1) * np.log(1 - x0 ** 2)

    for i in range(size):
        while True:
            z = np.random.beta((dim - 1) / 2, (dim - 1) / 2)
            u = np.random.rand()
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            if kappa * w + (dim - 1) * np.log(1 - x0 * w) - c >= np.log(u):
                break

        v = np.random.normal(size=dim - 1)
        v = v / np.linalg.norm(v)
        result[i, :-1] = np.sqrt(1 - w ** 2) * v
        result[i, -1] = w

    O = np.eye(dim)
    O[:, -1] = mu
    O[:, :-1] -= np.outer(mu, mu[:-1])
    O[:, :-1] /= np.linalg.norm(O[:, :-1], axis=0)
    result = np.dot(result, O.T)

    return result


# 三维情况下的 vMF 分布参数
mu = np.array([0, 0, 1])
kappa = 5
size = 1000

# 采样
samples = sample_vMF(mu, kappa, size)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D von Mises-Fisher Distribution Samples')
plt.show()
