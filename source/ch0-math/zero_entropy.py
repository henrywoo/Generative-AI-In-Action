import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

# Number of attention heads
num_heads = 30

# Uniform distribution
uniform_weights = np.ones(num_heads) / num_heads
uniform_entropy = entropy(uniform_weights, base=2)

# Gaussian distribution
gaussian_weights = np.random.normal(0.5, 0.1, num_heads)
gaussian_weights /= gaussian_weights.sum()  # Normalize
gaussian_entropy = entropy(gaussian_weights, base=2)

# One-hot distribution
onehot_weights = np.zeros(num_heads)
onehot_weights[0] = 1
onehot_entropy = entropy(onehot_weights, base=2)

# 3 Large Values distribution
three_large_weights = np.zeros(num_heads)
large_value_indices = np.random.choice(num_heads, 3, replace=False)  # Pick 3 indices randomly
three_large_weights[large_value_indices] = 0.9  # Set the chosen indices to 0.9
three_large_weights /= three_large_weights.sum()  # Normalize
three_large_entropy = entropy(three_large_weights, base=2)

# Create subplots (1 row, 4 columns)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plt.style.use("ggplot")

# Plot uniform distribution
axs[0].bar(range(num_heads), uniform_weights)
axs[0].set_title(f'Uniform (Entropy: {uniform_entropy:.2f})')
axs[0].set_xlabel('Attention Head')
axs[0].set_ylabel('Weight')


# Plot Gaussian distribution
axs[1].bar(range(num_heads), gaussian_weights)
axs[1].set_title(f'Gaussian (Entropy: {gaussian_entropy:.2f})')
axs[1].set_xlabel('Attention Head')
axs[1].set_ylabel('Weight')


# Plot one-hot distribution
axs[2].bar(range(num_heads), onehot_weights)
axs[2].set_title(f'One-Hot (Entropy: {onehot_entropy:.2f})')
axs[2].set_xlabel('Attention Head')
axs[2].set_ylabel('Weight')


# Plot 3 Large Values distribution
axs[3].bar(range(num_heads), three_large_weights)
axs[3].set_title(f'3 Large Values (Entropy: {three_large_entropy:.2f})')
axs[3].set_xlabel('Attention Head')
axs[3].set_ylabel('Weight')


# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('zero_entropy.png')
plt.show()
