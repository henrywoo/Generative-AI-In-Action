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

def gumbel_softmax(logits, temperature=1.0, eps=1e-20):
    """ Sample from the Gumbel-Softmax distribution and return samples in one-hot encoded form. """
    U = torch.rand_like(logits)
    G = -torch.log(-torch.log(U + eps) + eps)
    y = logits + G
    return torch.softmax(y / temperature, dim=-1)

# Define the number of classes and samples
num_classes = 10
num_samples = 1000

# Create logits for a categorical distribution over `num_classes` classes
logits = torch.randn(num_classes)

# Temperature parameter controls the discreteness
temperature = 0.5  # Try different values to see the effect (e.g., 0.1, 0.5, 1.0)

# Sample from Gumbel-Softmax
samples = torch.stack([gumbel_softmax(logits, temperature) for _ in range(num_samples)])

# Convert to numpy for plotting
samples_np = samples.numpy()

# Plotting
plt.figure(figsize=(12, 8))
for i in range(num_classes):
    plt.hist(samples_np[:, i], bins=50, alpha=0.6, label=f'Class {i+1}')

plt.title(f'Gumbel-Softmax Distribution (Temperature = {temperature})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig("gumbel.png")
plt.show()
