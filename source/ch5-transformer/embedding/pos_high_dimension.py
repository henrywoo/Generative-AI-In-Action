import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

def plot_vector(ax, vector, origin=[0, 0], color='r', label=None, linestyle='-'):
    ax.quiver(*origin, vector[0], vector[1], color=color, angles='xy', scale_units='xy', scale=1, linestyle=linestyle)
    if label:
        ax.text(vector[0] * 0.6, vector[1] * 0.6, label, fontsize=12, ha='center', color=color)

def plot_rotated_vector(ax, vector, angle, origin=[0, 0], color='b', label=None, linestyle='--'):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_vector = rotation_matrix @ vector
    plot_vector(ax, rotated_vector, origin, color=color, label=label, linestyle=linestyle)

# Original vectors in a 6D space, decomposed into three 2D vectors
vectors = [np.array([2, 0]), np.array([2, 1]), np.array([2, 2])]
angles = [np.pi / 6, np.pi / 4, np.pi / 3]  # 30, 45 , 60
colors = ['r', 'g', 'b']

fig, ax = plt.subplots(1, 2, figsize=(6, 3))

# Plot original vectors
ax[0].set_title('Original Vectors', fontsize=10)
for i, v in enumerate(vectors):
    plot_vector(ax[0], v, color=colors[i], label=f'v{i+1}', linestyle='-')

# Plot rotated vectors
ax[1].set_title('Rotated Vectors', fontsize=10)
for i, v in enumerate(vectors):
    plot_rotated_vector(ax[1], v, angles[i], color=colors[i], label=f'v{i+1}', linestyle='-')

for a in ax:
    a.set_xlim(-3, 4)
    a.set_ylim(-1, 4)
    a.grid(True)
    a.set_aspect('equal')

plt.tight_layout()
plt.savefig('pos_high_dimension.png')
plt.show()
