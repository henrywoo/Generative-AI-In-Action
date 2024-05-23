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
        u = np.random.uniform(0, M * proposal_dist(x))
        if u <= target_dist(x):
            samples.append(x)
        else:
            pass  # print("rejection")
    return np.array(samples)

# Parameters
n_samples = 50000
x_min, x_max = -6, 6
x = np.linspace(x_min, x_max, 5000)
# scaling factor (make sure M * proposal_dist >= target_dist for all x in [x_min, x_max])
M = np.max(target_distribution(x)) / (1/8) # (1/8) is the max value of proposal_dist

# Generate samples
samples = rejection_sampling(n_samples, target_distribution, proposal_distribution, M, x_min, x_max)

# Plotting
plt.figure(figsize=(8, 4))
plt.hist(samples, bins=500, density=True, alpha=0.6, color='g', label='Sampled Dist')
plt.plot(x, target_distribution(x), 'r', label='Target Dist(Complex)', alpha=0.5)
plt.plot(x, M * proposal_distribution(x), 'b--', label='Scaled Proposal Dist(Uniform)', alpha=0.5)
plt.legend()
plt.xlabel('x', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.title('Rejection Sampling', fontsize=10)
plt.savefig("rejection_sampling.png")
plt.show()
