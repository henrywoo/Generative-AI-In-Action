import numpy as np
import matplotlib.pyplot as plt

from rotary import scale_theta

# Parameters
theta = 2.0  # Initial theta value
positions = np.arange(1, 11)  # Positions from 1 to 10
embedding_dim = 2  # Embedding dimension

# Compute scaled theta values
scaled_thetas = [scale_theta(theta, pos, embedding_dim) for pos in positions]

# Plotting
plt.style.use('ggplot')
plt.figure(figsize=(8, 6))
plt.plot(positions, scaled_thetas, marker='o', linestyle='-', color='blue')
plt.title('Scaled Theta Values Across Positions')
plt.xlabel('Position')
plt.ylabel('Scaled Theta')
plt.grid(True)
plt.xticks(positions)  # Set x-ticks to show each position
plt.show()
