import torch
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# Set the style to 'ggplot' for aesthetic enhancements
plt.style.use('ggplot')

# Seed setting for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Sample data setup with a highly right-skewed random distribution
batch, sentence_length, embedding_dim = 100, 5, 10

# Create a highly right-skewed distribution for embeddings
# Using log-normal distribution to generate skewed data
mean, std_dev = 0, 0.8  # Parameters for the log-normal distribution
skewed_data = torch.empty(batch, sentence_length, embedding_dim).normal_(mean=mean, std=std_dev).exp_()

# Initialize LayerNorm
layer_norm = nn.LayerNorm(embedding_dim)

# Apply Layer Normalization
normalized_embedding = layer_norm(skewed_data)

# Convert tensors to numpy for plotting by detaching from the computation graph
embedding_np = skewed_data.detach().numpy()
normalized_embedding_np = normalized_embedding.detach().numpy()

# Flatten the tensors to 1D for simpler histogram visualization
embedding_flat = embedding_np.flatten()
normalized_embedding_flat = normalized_embedding_np.flatten()

# Calculate statistics
mean_orig, std_orig = embedding_flat.mean(), embedding_flat.std()
kurtosis_orig = kurtosis(embedding_flat)
mean_norm, std_norm = normalized_embedding_flat.mean(), normalized_embedding_flat.std()
kurtosis_norm = kurtosis(normalized_embedding_flat)

# Create histograms to compare the embeddings before and after Layer Normalization
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].hist(embedding_flat, bins=60, color='blue', alpha=0.7, label='Original Embeddings')
ax[0].set_title('Histogram of Original Embeddings')
ax[0].set_xlabel('Value')
ax[0].set_ylabel('Frequency')
ax[0].grid(True)  # Enable grid
stats_text_orig = f'Mean: {mean_orig:.2f}, Std: {std_orig:.2f}, Kurtosis: {kurtosis_orig:.2f}'
ax[0].text(0.95, 0.95, stats_text_orig, transform=ax[0].transAxes, horizontalalignment='right', verticalalignment='top', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

ax[1].hist(normalized_embedding_flat, bins=60, color='green', alpha=0.7, label='Normalized Embeddings')
ax[1].set_title('Histogram of Normalized Embeddings')
ax[1].set_xlabel('Value')
ax[1].set_ylabel('Frequency')
ax[1].grid(True)  # Enable grid
stats_text_norm = f'Mean: {mean_norm:.2f}, Std: {std_norm:.2f}, Kurtosis: {kurtosis_norm:.2f}'
ax[1].text(0.95, 0.95, stats_text_norm, transform=ax[1].transAxes, horizontalalignment='right', verticalalignment='top', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

# Adjust layout and plot
plt.tight_layout()
plt.savefig("layernorm_demo_v2.png")
plt.show()
