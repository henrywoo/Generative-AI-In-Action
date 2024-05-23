import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Define the target distribution (mixture of two Gaussians)
def target_distribution(x):
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + 0.7 * np.exp(-0.5 * ((x + 2) / 1.0)**2)

# Define the proposal distribution (uniform distribution)
def proposal_distribution(x):
    return np.ones_like(x) / 8

# Rejection sampling algorithm
def rejection_sampling(n_samples, target_dist, proposal_dist, M, x_min, x_max):
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(x_min, x_max)
        u = np.random.uniform(0, M * proposal_dist(np.array([x])))
        if u <= target_dist(np.array([x])):
            samples.append(x)
    return np.array(samples)

# Parameters
n_samples = 5000
x_min, x_max = -6, 6
M = 1.5  # scaling factor (make sure M * proposal_dist >= target_dist for all x in [x_min, x_max])

# Generate samples
samples = rejection_sampling(n_samples, target_distribution, proposal_distribution, M, x_min, x_max)

# Plotting
x = np.linspace(x_min, x_max, 1000)
plt.figure(figsize=(6, 3))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Sampled Distribution')
plt.plot(x, target_distribution(x), 'r', label='Target Distribution', alpha=0.5)
plt.plot(x, M * proposal_distribution(x), 'b--', label='Scaled Proposal Distribution', alpha=0.5)
plt.legend()
plt.xlabel('x', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.title('Rejection Sampling', fontsize=10)
plt.savefig("rejection_sampling.png")
plt.show()
