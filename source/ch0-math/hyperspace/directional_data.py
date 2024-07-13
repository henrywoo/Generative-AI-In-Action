import numpy as np
import matplotlib.pyplot as plt

# 生成随机方向性数据（单位圆上的点）
num_points = 100
angles = np.random.uniform(0, 2 * np.pi, num_points)
x = np.cos(angles)
y = np.sin(angles)

# 可视化方向性数据
plt.figure(figsize=(6, 6))
plt.quiver(np.zeros(num_points), np.zeros(num_points), x, y, angles, scale=10, scale_units='xy', angles='xy', cmap='hsv')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Directional Data')
plt.grid()
plt.show()
