import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the original multi-variable function
def f(x, y):
    return np.sin(x) + np.cos(y)

# Define functions phi and psi as single-variable functions
def phi_i(z):
    return np.sin(z)

def psi_j(x):
    return np.cos(x)

# Define the representation using the theorem
def kolmogorov_arnold(x, y, n_terms):
    result = 0
    for i in range(n_terms*2+1):
        intermediate_sum = 0
        for j in range(n_terms):
            intermediate_sum += psi_j(x + j) + psi_j(y + j)
        result += phi_i(intermediate_sum + i)
    return result

# Create a grid of points for plotting
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z_original = f(X, Y)

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(6, 6))

def update(n_terms):
    ax.clear()
    Z_approx = kolmogorov_arnold(X, Y, n_terms)
    contour = ax.contourf(X, Y, Z_approx, cmap='Reds')
    ax.set_title(f'Approximation using Kolmogorov-Arnold Theorem\nn_terms={n_terms}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return contour.collections

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(5, 51), repeat=False)

# Save animation as mp4
ani.save('kolmogorov_arnold_approximation.mp4', writer='ffmpeg')

plt.close(fig)
