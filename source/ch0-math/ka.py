import numpy as np
import matplotlib.pyplot as plt

# Define the original multi-variable function
def f(x, y):
    return np.sin(x) + np.cos(y)

# Define more complex functions phi and psi as single-variable functions
def phi_i(z):
    return np.sin(z)  # A function of a single variable

def psi_j(x):
    return np.cos(x)  # A function of a single variable

# Define the representation using the theorem
def kolmogorov_arnold(x, y, n_terms):
    result = 0
    for i in range(n_terms):
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
Z_approx = kolmogorov_arnold(X, Y, n_terms=20)

# Plot the original function
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z_original, cmap='viridis')
plt.colorbar()
plt.title('Original Function $f(x, y) = \sin(x) + \cos(y)$')
plt.xlabel('$x$')
plt.ylabel('$y$')

# Plot the approximation
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z_approx, cmap='viridis')
plt.colorbar()
plt.title('Approximation using Kolmogorov-Arnold Theorem')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.tight_layout()
plt.show()
