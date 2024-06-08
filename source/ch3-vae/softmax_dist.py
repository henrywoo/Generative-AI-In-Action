import numpy as np
import matplotlib.pyplot as plt

logits = np.random.normal(loc=0, scale=1, size=5000)  # Sample logits from a normal distribution
probs = np.exp(logits) / np.sum(np.exp(logits))        # Apply softmax

plt.hist(probs, bins=100, density=True, alpha=0.6, color='skyblue')
plt.title('Softmax Probabilities from Normally Distributed Logits')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.show()
