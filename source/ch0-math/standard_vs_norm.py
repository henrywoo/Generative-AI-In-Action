import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


plt.style.use('ggplot')

# Seed setting for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Generate sample data
np.random.seed(0)  # For reproducibility
data = torch.tensor(np.random.randint(-100, 100, (100, 4)).astype(float))  # Larger range and more data points

# Initialize scalers
standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

# Fit and transform the data
standardized_data = standard_scaler.fit_transform(data)
normalized_data = minmax_scaler.fit_transform(data)

# Visualization
fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # Increase subplot grid to 2x3

# Scatter plots
# Original Data
axs[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.5)
axs[0, 0].set_title('Original Data')
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')

# Standardized Data
axs[0, 1].scatter(standardized_data[:, 0], standardized_data[:, 1], alpha=0.5)
axs[0, 1].set_title('Standardized Data')
axs[0, 1].set_xlabel('Feature 1')
axs[0, 1].set_ylabel('Feature 2')

# Normalized Data
axs[0, 2].scatter(normalized_data[:, 0], normalized_data[:, 1], alpha=0.5)
axs[0, 2].set_title('Normalized Data')
axs[0, 2].set_xlabel('Feature 1')
axs[0, 2].set_ylabel('Feature 2')

# Histograms
# Original Data Histogram
axs[1, 0].hist(data.numpy().ravel(), bins=20, color='blue', alpha=0.7)
axs[1, 0].set_title('Histogram of Original Data')

# Standardized Data Histogram
axs[1, 1].hist(standardized_data.ravel(), bins=20, color='orange', alpha=0.7)
axs[1, 1].set_title('Histogram of Standardized Data')

# Normalized Data Histogram
axs[1, 2].hist(normalized_data.ravel(), bins=20, color='green', alpha=0.7)
axs[1, 2].set_title('Histogram of Normalized Data')

plt.tight_layout()
plt.savefig("standard_vs_norm.png")
plt.show()

# Print statistics to show effects
print("Original Data Stats:")
print(f"Mean: {data.float().mean(dim=0)}")
print(f"Standard Deviation: {data.std(dim=0)}")

print("\nStandardized Data Stats:")
print(f"Mean: {standardized_data.mean(axis=0)}")
print(f"Standard Deviation: {standardized_data.std(axis=0)}")

print("\nNormalized Data Stats:")
print(f"Min: {normalized_data.min(axis=0)}")
print(f"Max: {normalized_data.max(axis=0)}")
