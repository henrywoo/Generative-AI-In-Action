import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, gamma
import numpy as np
import random

plt.style.use('ggplot')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define the input vector with left-skewed distribution
# Using gamma distribution with a shape parameter greater than 1
gamma_shape = 2  # Shape parameter
gamma_scale = 50  # Scale parameter (to adjust range of values)
vector = torch.tensor(gamma.rvs(gamma_shape, scale=gamma_scale, size=100)) * 20 - 10  # Adjusting range

# Initialize LayerNorm
layer_norm = torch.nn.LayerNorm(normalized_shape=vector.size(), elementwise_affine=False)

# Apply Layer Normalization
normalized_vector = layer_norm(vector)

# Convert to numpy for plotting
vector_np = vector.numpy()
normalized_vector_np = normalized_vector.numpy()

# Calculate statistics
mean_original = vector_np.mean()
std_original = vector_np.std()
kurtosis_original = kurtosis(vector_np)

mean_normalized = normalized_vector_np.mean()
std_normalized = normalized_vector_np.std()
kurtosis_normalized = kurtosis(normalized_vector_np)

# Plot the vectors with horizontal lines and histograms
fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # Adjusted subplot grid to 2x2

# Bar plots
# Original Vector
ax[0, 0].bar(range(len(vector_np)), vector_np, color='blue')
ax[0, 0].set_title('Original Vector (Left-Skewed)')
ax[0, 0].set_xlabel('Index')
ax[0, 0].set_ylabel('Value')

# Layer Normalized Vector
ax[1, 0].bar(range(len(normalized_vector_np)), normalized_vector_np, color='green')
ax[1, 0].set_title('Layer Normalized Vector')
ax[1, 0].set_xlabel('Index')
ax[1, 0].set_ylabel('Value')

# Histograms
# Histogram for Original Vector
ax[0, 1].hist(vector_np, bins=40, color='blue', alpha=0.7)
ax[0, 1].set_title('Distribution of Original Vector (Left-Skewed)')
ax[0, 1].set_xlabel('Value')
ax[0, 1].set_ylabel('Frequency')

# Histogram for Normalized Vector
ax[1, 1].hist(normalized_vector_np, bins=40, color='green', alpha=0.7)
ax[1, 1].set_title('Distribution of Normalized Vector')
ax[1, 1].set_xlabel('Value')
ax[1, 1].set_ylabel('Frequency')

# Display statistics
# Original Vector
stats_text_original = f'Mean: {mean_original:.2f}\nSTD: {std_original:.2f}\nKurtosis: {kurtosis_original:.2f}'
ax[0, 0].text(0.5, 0.95, stats_text_original, transform=ax[0, 0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle="round", alpha=0.5))

# Normalized Vector
stats_text_normalized = f'Mean: {mean_normalized:.2f}\nSTD: {std_normalized:.2f}\nKurtosis: {kurtosis_normalized:.2f}'
ax[1, 0].text(0.5, 0.95, stats_text_normalized, transform=ax[1, 0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle="round", alpha=0.5))

# Adjust layout and save
plt.tight_layout()
plt.savefig("layernorm_demo_v1.png")
plt.show()
