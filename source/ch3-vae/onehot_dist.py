import matplotlib.pyplot as plt
import numpy as np

# Define one-hot vectors
one_hot_3 = [1, 0, 0]
one_hot_5 = [0, 0, 1, 0, 0]
one_hot_8 = [0, 0, 0, 0, 0, 1, 0, 0]

plt.style.use('ggplot')
# Plotting
fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))  # 1 row, 3 columns for subplots

axs[0].bar(range(len(one_hot_3)), one_hot_3)
axs[0].set_title('One-Hot Dist(L=3)', fontsize=9)
axs[0].set_xticks(range(len(one_hot_3)))
axs[0].set_xlabel('Category', fontsize=8)
axs[0].set_ylabel('Probability', fontsize=8)

axs[1].bar(range(len(one_hot_5)), one_hot_5)
axs[1].set_title('One-Hot Dist(L=5)', fontsize=9)
axs[1].set_xticks(range(len(one_hot_5)))
axs[1].set_xlabel('Category', fontsize=8)
axs[1].set_ylabel('Probability', fontsize=8)

axs[2].bar(range(len(one_hot_8)), one_hot_8)
axs[2].set_title('One-Hot Dist(L=8)', fontsize=9)
axs[2].set_xticks(range(len(one_hot_8)))
axs[2].set_xlabel('Category', fontsize=8)
axs[2].set_ylabel('Probability', fontsize=8)

plt.tight_layout()
plt.grid(True)
plt.savefig("one_hot_dist.png")
plt.show()
