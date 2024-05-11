# https://omkar-ranadive.github.io/posts/stats-IS
# Sampling is actually a misnomer. Using Importance Sampling we are essentially approximating the expected value of
# some distribution p(X) using another distribution q(X).
import numpy as np
import matplotlib.pyplot as plt

# Set the ggplot style
plt.style.use('ggplot')

# Parameters for the distributions
target_mean = 5
target_std = 1
proposal_mean = 0
proposal_std = 2

# Function to calculate the probability density of a normal distribution
def normal_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

# Number of samples
N = 1000

# Generate samples from the proposal distribution
samples = np.random.normal(proposal_mean, proposal_std, N)

# Calculate weights for importance sampling
weights = normal_pdf(samples, target_mean, target_std) / normal_pdf(samples, proposal_mean, proposal_std)
normalized_weights = weights / np.sum(weights)

# Estimate the mean of the target distribution
estimated_mean = np.sum(samples * normalized_weights)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=30, alpha=0.6, label='Samples from Proposal Distribution')
plt.axvline(x=target_mean, color='r', linestyle='dashed', linewidth=2, label='Target Mean')
plt.axvline(x=estimated_mean, color='g', linestyle='dashed', linewidth=2, label='Estimated Mean')
plt.title(f'Importance Sampling Estimation\nEstimated Mean: {estimated_mean:.2f}')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('importance_sampling.png')
plt.show()
