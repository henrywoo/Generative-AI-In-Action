import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10  # Number of sequences
T = 5    # Average length of each sequence

# Simulating log probabilities of actions from a policy model
# Assuming log probabilities range from -2 to 0 (more negative is less probable)
log_probs = np.random.uniform(-2, 0, size=(N, T))

# Simulating cumulative rewards for each sequence
# Rewards could range from -3 to 3 for example
rewards = np.random.uniform(-3, 3, size=(N))

# Weighted sum of log probabilities by rewards
weighted_log_probs = np.array([log_probs[n] * rewards[n] for n in range(N)])

# Compute loss for each sequence
losses = -np.mean(np.sum(weighted_log_probs, axis=1))

# Plotting
plt.style.use('ggplot')
plt.figure(figsize=(10, 10))
for n in range(N):
    plt.plot(weighted_log_probs[n], label=f'Sequence {n+1} Reward: {rewards[n]:.2f}', marker='o', alpha=0.5)
plt.title(f'Impact of Rewards on Log Probabilities (Policy Gradient Updates) | Loss: {losses:.2f}')
plt.xlabel('Time Step')
plt.ylabel('Weighted Log Probability')
plt.legend()
plt.grid(True)
plt.show()
