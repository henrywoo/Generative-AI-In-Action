import numpy as np
import matplotlib.pyplot as plt
from hiq import deterministic


# 定义一个简单的二元高斯分布的对数似然函数
def log_likelihood(mu, sigma, x):
    return -0.5 * np.log(2 * np.pi * sigma ** 2) - (x - mu) ** 2 / (2 * sigma ** 2)


# 计算对数似然函数的一阶导数（score function）
def score(mu, sigma, x):
    dL_dmu1 = (x[0] - mu[0]) / sigma ** 2
    dL_dmu2 = (x[1] - mu[1]) / sigma ** 2
    dL_dsigma = ((x[0] - mu[0]) ** 2 + (x[1] - mu[1]) ** 2 - 2 * sigma ** 2) / sigma ** 3
    return np.array([dL_dmu1, dL_dmu2, dL_dsigma])


# 计算 Fisher 信息矩阵
def fisher_information(mu, sigma, x):
    n = len(x)
    scores = np.array([score(mu, sigma, xi) for xi in x])
    fisher_info = np.zeros((3, 3))  # 初始化 Fisher 信息矩阵
    for i in range(n):
        fisher_info += np.outer(scores[i], scores[i])
    fisher_info /= n
    return fisher_info


# 生成样本数据
np.random.seed(0)
mu_true = np.array([0.0, 0.0])
sigma_true = 1.0
x = np.random.multivariate_normal(mu_true, np.eye(2) * sigma_true, size=1000)

# 计算 Fisher 信息矩阵
mu_est = np.mean(x, axis=0)
sigma_est = np.std(x)
fim = fisher_information(mu_est, sigma_est, x)

print("Fisher Information Matrix:")
print(fim)


# 普通梯度下降
def gradient_descent(mu, sigma, data, learning_rate, num_steps):
    mu_trajectory = [mu.copy()]
    sigma_trajectory = [sigma]
    for step in range(num_steps):
        grad_mu1 = np.mean([(xi[0] - mu[0]) / sigma ** 2 for xi in data], axis=0)
        grad_mu2 = np.mean([(xi[1] - mu[1]) / sigma ** 2 for xi in data], axis=0)
        grad_sigma = np.mean(
            [((xi[0] - mu[0]) ** 2 + (xi[1] - mu[1]) ** 2 - 2 * sigma ** 2) / sigma ** 3 for xi in data])

        mu -= learning_rate * np.array([grad_mu1, grad_mu2])
        sigma -= learning_rate * grad_sigma

        mu_trajectory.append(mu.copy())
        sigma_trajectory.append(sigma)

    return mu_trajectory, sigma_trajectory


# 自然梯度下降
def natural_gradient_descent(mu, sigma, data, learning_rate, num_steps):
    mu_trajectory = [mu.copy()]
    sigma_trajectory = [sigma]
    for step in range(num_steps):
        grad_mu1 = np.mean([(xi[0] - mu[0]) / sigma ** 2 for xi in data], axis=0)
        grad_mu2 = np.mean([(xi[1] - mu[1]) / sigma ** 2 for xi in data], axis=0)
        grad_sigma = np.mean(
            [((xi[0] - mu[0]) ** 2 + (xi[1] - mu[1]) ** 2 - 2 * sigma ** 2) / sigma ** 3 for xi in data])

        fisher_inv = np.linalg.inv(fisher_information(mu, sigma, data))
        natural_grad = np.array([grad_mu1, grad_mu2, grad_sigma])

        update = learning_rate * fisher_inv @ natural_grad

        mu -= update[:2]
        sigma -= update[2]

        # 确保 sigma 不变得太小
        sigma = max(sigma, 0.1)

        mu_trajectory.append(mu.copy())
        sigma_trajectory.append(sigma)

    return mu_trajectory, sigma_trajectory


# 初始点
start_mu = np.array([2.0, 2.0])
start_sigma = 2.0

# 学习率和步数
learning_rate = 0.01  # 调小学习率
num_steps = 50  # 增加步数

# 运行普通梯度下降和自然梯度下降
mu_traj_gd, sigma_traj_gd = gradient_descent(start_mu.copy(), start_sigma, x, learning_rate, num_steps)
mu_traj_ngd, sigma_traj_ngd = natural_gradient_descent(start_mu.copy(), start_sigma, x, learning_rate, num_steps)

# 绘制优化轨迹
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))

# 绘制 mu 的轨迹
plt.subplot(1, 2, 1)
mu_traj_gd = np.array(mu_traj_gd)
mu_traj_ngd = np.array(mu_traj_ngd)
plt.plot(mu_traj_gd[:, 0], mu_traj_gd[:, 1], label='Gradient Descent', linewidth=5, alpha=0.7)
plt.plot(mu_traj_ngd[:, 0], mu_traj_ngd[:, 1], label='Natural Gradient Descent', linewidth=2, alpha=0.7)
plt.scatter([start_mu[0]], [start_mu[1]], c='red', label='Start')
plt.title('Trajectory of mu')
plt.xlabel('mu_1')
plt.ylabel('mu_2')
plt.legend()

# 绘制 sigma 的轨迹
plt.subplot(1, 2, 2)
plt.plot(sigma_traj_gd, label='Gradient Descent', linewidth=5, alpha=0.7)
plt.plot(sigma_traj_ngd, label='Natural Gradient Descent', linewidth=2, alpha=0.7)
plt.scatter([0], [start_sigma], c='red', label='Start')
plt.title('Trajectory of sigma')
plt.xlabel('Step')
plt.ylabel('sigma')
plt.legend()

plt.tight_layout()
plt.savefig("fisher2.png")
plt.show()
