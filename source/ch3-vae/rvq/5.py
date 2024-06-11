import matplotlib.pyplot as plt

from points import *



# Make the nested codebooks
n_codebooks = 3
codebook = generate_codebook(n_grid)
codebooks = [codebook/n_grid**level for level in range(n_codebooks)]

indices = quantizer(data, codebooks)   # call the quantizer
K = n_codebooks
fig, ax_all = plt.subplots(nrows=1, ncols=K, figsize=(10 + (K - 3), 3))
recon = data * 0
for axi in range(K):
    ax = ax_all[axi]
    ax.scatter(data[:, 0], data[:, 1], s=16)
    recon += codebooks[axi][indices[axi]]
    ax.scatter(recon[:, 0], recon[:, 1], s=16, color='orange')
    ax.set_xticks([])
    ax.set_yticks([])
    hpos = 1 / K * 0.85 + 1 / K * 0.81 * axi if K == 4 else .25 + .27 * axi
    fig.text(hpos, .05, f"{axi + 1} codebook{'s' if axi > 0 else ''}", ha='center')
    error = ((recon - data) ** 2).mean()
    fig.text(hpos, .0001, f"Error = {error:.2e}", ha='center')
plt.grid(True)
plt.show()