import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

def volume_n_sphere(n, r=1):
    """计算n维球的体积"""
    return (np.pi ** (n / 2) / gamma(n / 2 + 1)) * (r ** n)

def surface_area_n_sphere(n, r=1):
    """计算n维球的表面积"""
    return (2 * np.pi ** ((n + 1) / 2) / gamma((n + 1) / 2)) * (r ** n)

# 维度范围
dimensions = np.arange(1, 21)

# 计算体积和表面积
volumes = [volume_n_sphere(n) for n in dimensions]
surface_areas = [surface_area_n_sphere(n) for n in dimensions]

# 创建图形
plt.style.use('ggplot')
fig, ax1 = plt.subplots(1, 2, figsize=(8, 4))

# 绘制体积图
ax1[0].plot(dimensions, volumes, 'bo-', label='Volume')
ax1[0].set_xlabel('Dimension', fontsize=8)
ax1[0].set_ylabel('Volume', fontsize=8)
ax1[0].set_title('Dimension vs. Volume of n-Sphere', fontsize=9)
ax1[0].legend()
ax1[0].grid(True)

# 绘制表面积图
ax1[1].plot(dimensions, surface_areas, 'ro-', label='Surface Area')
ax1[1].set_xlabel('Dimension', fontsize=8)
ax1[1].set_ylabel('Surface Area', fontsize=8)
ax1[1].set_title('Dimension vs. Surface Area of n-Sphere', fontsize=9)
ax1[1].legend()
ax1[1].grid(True)

# 显示图形
plt.tight_layout()
plt.savefig('hyperball_d_vs_vol_surface.png')
plt.show()
