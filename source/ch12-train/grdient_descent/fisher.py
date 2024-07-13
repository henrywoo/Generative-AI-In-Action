import numpy as np
import matplotlib.pyplot as plt
from hiq import deterministic

# 定义一个简单的二次函数
def f(x):
    return 0.5 * (x[0]**2 + 10 * x[1]**2)

# 计算梯度
def grad_f(x):
    return np.array([x[0], 10 * x[1]])

# Fisher 信息矩阵（假设为常数矩阵）
def fisher_information(x):
    return np.array([[1, 0], [0, 10]])

# 普通梯度下降
def gradient_descent(start, learning_rate, num_steps):
    x = start
    trajectory = [x]
    for _ in range(num_steps):
        x = x - learning_rate * grad_f(x)
        trajectory.append(x)
    return np.array(trajectory)

# 自然梯度下降
def natural_gradient_descent(start, learning_rate, num_steps):
    x = start
    trajectory = [x]
    for _ in range(num_steps):
        fisher_inv = np.linalg.inv(fisher_information(x))
        natural_grad = fisher_inv @ grad_f(x)
        x = x - learning_rate * natural_grad
        trajectory.append(x)
    return np.array(trajectory)

# 初始点
start = np.array([10.0, 10.0])

# 学习率和步数
learning_rate = 0.1
num_steps = 50

# 运行梯度下降和自然梯度下降
trajectory_gd = gradient_descent(start, learning_rate, num_steps)
trajectory_ngd = natural_gradient_descent(start, learning_rate, num_steps)

# 绘制优化轨迹
plt.style.use('ggplot')
plt.plot(trajectory_gd[:, 0], trajectory_gd[:, 1], label='Gradient Descent')
plt.plot(trajectory_ngd[:, 0], trajectory_ngd[:, 1], label='Natural Gradient Descent')
plt.scatter([start[0]], [start[1]], c='red', label='Start')
plt.title('Gradient Descent vs. Natural Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('fisher.png')
plt.show()
