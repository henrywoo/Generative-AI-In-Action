import numpy as np
import matplotlib.pyplot as plt

def gaussian_to_circle(data):
    transformed_data = []
    for point in data:
        z_norm = np.linalg.norm(point)
        transformed_point = point / 10 + point / z_norm
        transformed_data.append(transformed_point)
    return np.array(transformed_data)

# Generate 2D Gaussian data
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # Identity matrix for standard Gaussian
data = np.random.multivariate_normal(mean, cov, 500)

# Transform the data
transformed_data = gaussian_to_circle(data)

# Plot original Gaussian data
plt.style.use('ggplot')
plt.figure(figsize=(8, 3.8))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
plt.title('Original Gaussian Data', fontsize=9)
plt.xlabel('X')
plt.ylabel('Y')

# Plot transformed circular data
plt.subplot(1, 2, 2)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
plt.title('Transformed Circular Data', fontsize=9)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('gaussian_to_circle.png')
plt.show()
