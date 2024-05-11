import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def cross_entropy(p, q):
    return -np.sum(p * np.log(q))

# True distribution P
P = np.array([0.1, 0.2, 0.3, 0.4])

# Q distribution will vary
Q_variations = np.linspace(0.01, 1, 100)
Q_matrix = np.array([Q_variations, 1-Q_variations, Q_variations, 1-Q_variations]).T
Q_matrix /= Q_matrix.sum(axis=1, keepdims=True)

# Calculating metrics
kl = [kl_divergence(P, Q) for Q in Q_matrix]
ce = [cross_entropy(P, Q) for Q in Q_matrix]

# Plotting
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(Q_variations, kl, label='KL Divergence')
plt.plot(Q_variations, ce, label='Cross-Entropy')
plt.title('KL Divergence vs Cross-Entropy')
plt.xlabel('Variation in Q')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.savefig('kl_vs_ce.png')
plt.show()
