import numpy as np
import matplotlib.pyplot as plt

# Define the original multi-variable function
def f(x, y):
    return np.sin(x) + np.cos(y)

# Define single-variable functions phi and psi
def phi_q(z):
    return np.sin(z)

def psi_pq(x):
    return np.cos(x)

# Define the representation using the theorem
def kolmogorov_arnold(x, y, n):
    result = 0
    for q in range(1, 2*n + 2):
        intermediate_sum = 0
        for p in range(1, n + 1):
            intermediate_sum += psi_pq(x + p + q) + psi_pq(y + p + q)
        result += phi_q(intermediate_sum)
    return result

# Create a grid of points for plotting
x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
X, Y = np.meshgrid(x, y)
Z_original = f(X, Y)
Z_approx = kolmogorov_arnold(X, Y, n=20)

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
plt.savefig('kolmogorov_arnold_1.png')
plt.show()
