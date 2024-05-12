import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


d = 12
positions = np.arange(1, 21)
dimensions = np.arange(0, d, 2)

# Create meshgrid for positions and dimensions
X, Y = np.meshgrid(positions, dimensions)

def rope_sinusoid(position, i, d):
    """Calculates the sinusoidal value for a given position and dimension."""
    exponent = -2 * (i // 2) / d
    theta = position / (10000 ** exponent)
    return np.sin(theta) if i % 2 == 0 else np.cos(theta)

# Calculate Z values for each position and dimension
Z = np.array([[rope_sinusoid(pos, i, d) for pos in positions] for i in dimensions])

# Create 3D plot
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add colorbar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Customize plot
ax.set_xlabel('Position')
ax.set_ylabel('Dimension Index (i)')
ax.set_zlabel('Sinusoidal Value')
ax.set_title(f"Sinusoidal Positional Encoding (d={d})")

plt.show()
