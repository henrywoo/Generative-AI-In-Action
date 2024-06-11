from vector_quantize_pytorch import ResidualVQ
from points import *
import torch
import os

torch.manual_seed(0)
def make_pretty(data, d, cb_lengths, K_choices, raw_data):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', cmap='plasma', vmin=-12, vmax=1)
    plt.grid(False)
    plt.colorbar(label='log10(Error)')
    plt.xticks(ticks=np.arange(len(K_choices)), labels=K_choices)
    plt.yticks(ticks=np.arange(len(cb_lengths)), labels=cb_lengths)
    plt.xlabel('K choices')
    plt.ylabel('Codebook lengths')
    plt.title(f'd = {d}')
    # Annotate each cell with the error value
    for i in range(len(cb_lengths)):
        for j in range(len(K_choices)):
            plt.text(j, i, f'{10 ** raw_data[i, j]:.2e}', ha='center', va='center', color='white')
    # Save the image
    if not os.path.exists('images'):
        os.makedirs('images')
    plt.savefig(f'images/d_{d}_k_{len(K_choices)}.png')
    plt.close()


def show_colored_data(results_d, d, cb_lengths, K_choices):
    "displays a heatmap but colors the background according to log10 of the numbers"
    rdf = np.log10(results_d.numpy())
    make_pretty(rdf, d, cb_lengths, K_choices, results_d.numpy())
    return


n_dim = 256  # number of dimensions
cb_len = 256  # codebook length
K = 4  # number of codebooks
npoints_hd = 4096  # number of data points in high-dim space

d_choices = [2, 3, 6, 8, 16, 32, 64, 128, 256, 512]  # dimensions to try
cb_lengths = [25, 64, 256, 1024, 2048]  # codebook lengths
K_choices = [1, 2, 3, 4, 6, 8, 10]  # variable numbers of codebooks

results = torch.empty((len(d_choices), len(cb_lengths), len(K_choices))).cpu()
for q1, n_dim in enumerate(d_choices):
    for q2, cb_len in enumerate(cb_lengths):
        for q3, K in enumerate(K_choices):
            residual_vq = ResidualVQ(
                dim=n_dim,
                codebook_size=cb_len,
                num_quantizers=K,
                kmeans_init=True,  # set to True
                kmeans_iters=10  # number of kmeans iterations to calculate the centroids for the codebook on init
            )
            x = torch.randn(1, npoints_hd, n_dim)
            quantized, indices, commit_loss = residual_vq(x)
            error = ((quantized - x) ** 2).mean()
            results[q1, q2, q3] = error
    show_colored_data(results[q1], n_dim, cb_lengths, K_choices)

fig, ax = plt.subplots(figsize=(6.5, 4))
ourdata = results[:, -1, :]  # max cb_len
for q1, d in enumerate(d_choices):
    x = np.array(K_choices)
    y = ourdata[q1]
    ax.semilogy(x, y, 'o-', label=f"d = {d}", alpha=0.5)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlabel("K")
ax.set_ylabel("error")
plt.show()
