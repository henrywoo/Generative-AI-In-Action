from math import log2
from matplotlib import pyplot

# list of probabilities
probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# calculate information
info = [-log2(p) for p in probs]

# calculate entropy
def entropy(events, ets=1e-15):
    return -sum([p * log2(p + ets) for p in events])

# define probabilities
probs_entropy = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# create probability distribution
dists = [[p, 1.0 - p] for p in probs_entropy]

# calculate entropy for each distribution
ents = [entropy(d) for d in dists]

# calculate cross-entropy
def cross_entropy(p, q, ets=1e-15):
    return -sum([p_i * log2(q_i + ets) for p_i, q_i in zip(p, q)])

# create subplots
fig, axs = pyplot.subplots(1, 2, figsize=(10, 3))

# plot probability vs information
axs[0].plot(probs, info, marker='.')
axs[0].set_title('Probability vs Information')
axs[0].set_xlabel('Event Probability')
axs[0].set_ylabel('Information')

# plot probability distribution vs entropy
axs[1].plot(probs_entropy, ents, marker='.')
axs[1].set_title('Probability Distribution vs Entropy')
axs[1].set_xticks(probs_entropy)
axs[1].set_xticklabels([str(d) for d in dists])
axs[1].set_xlabel('RV Probability Distribution')
axs[1].set_ylabel('Entropy (bits)')


# adjust layout
pyplot.tight_layout()
pyplot.show()
