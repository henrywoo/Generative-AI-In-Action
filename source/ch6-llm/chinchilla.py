import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants from the Chinchilla scaling law
E = 1.69
A = 406.4
B = 410.7
alpha = 0.34
beta = 0.28

# Generating data: model sizes N and data sizes D
N = np.linspace(1, 10000, 100)  # Model size from 1 to 10,000
D = np.linspace(1, 10000, 100)  # Data size from 1 to 10,000
N, D = np.meshgrid(N, D)
L = E + A * N**alpha + B * D**beta  # Computing the loss using the scaling law

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(N, D, L, cmap='viridis')

# Labels and title
ax.set_xlabel('Model Size (N)')
ax.set_ylabel('Data Size (D)')
ax.set_zlabel('Loss (L)')
ax.set_title('Chinchilla Scaling Law')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Displaying parameter info on the plot
param_text = f'E = {E}, A = {A}, B = {B}, alpha = {alpha}, beta = {beta}'
ax.text2D(0.05, 0.01, param_text, transform=ax.transAxes)

plt.show()
