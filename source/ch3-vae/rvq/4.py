from matplotlib import collections as mc
from points import *

def plot_data_grid_with_resids(data, n_grid=5, hide_tick_labels=True, show_indices=False, show_centroids=False,
                               show_next_level_grid=False, show_resids=True, codebook=None, show_grid=True):
    "big ol' workhorse plotting routine that we'll progressively make use of as the lesson proceeds"
    fig, ax = plt.subplots(figsize=(3, 3))
    h = 1.0 / n_grid
    ax.set_xlim(DATA_MIN, DATA_MAX)
    ax.set_ylim(DATA_MIN, DATA_MAX)

    if show_grid:
        for i in range(n_grid + 1):
            ax.axhline(DATA_MIN + i * h, color='black')
            ax.axvline(DATA_MIN + i * h, color='black')

    if show_next_level_grid:  # draws lines in the middle
        x_start = 2 * h
        y_start = -h / 2
        for i in range(n_grid):  # horizontal lines
            y = y_start + i * h / n_grid
            ax.axhline(y, xmin=x_start, xmax=x_start + h, color='black')
        y_start, x_start = x_start, y_start
        for j in range(n_grid):  # horizontal lines
            x = x_start + j * h / n_grid
            ax.axvline(x, ymin=y_start, ymax=y_start + h, color='black')

    if show_indices:
        index = 0
        for j in range(n_grid):
            for i in range(n_grid):
                x = DATA_MIN + (i + 0.5) / n_grid
                y = DATA_MIN + 1 - (j + 0.5) / n_grid
                ax.text(x, y, str(index), ha='center', va='center', fontsize=14)
                index += 1

    if show_centroids:
        for j in range(n_grid):
            for i in range(n_grid):
                x = DATA_MIN + (i + 0.5) * h
                y = DATA_MIN + (j + 0.5) * h
                ax.plot(x, y, 'bv', markersize=6)

    if show_resids and codebook is not None:
        memb = get_region_membership(data, codebook=codebook)
        resids = data * 0
        lines = []
        for i, p in enumerate(data):
            # resids[i] = p - codebook[memb[i]] # don't actually need to compute resids for this
            lines.append([p, codebook[memb[i]]])
        lc = mc.LineCollection(lines, colors=(1, 0, 1, 1), linewidths=1, linestyles='--', alpha=0.5)
        ax.add_collection(lc)

    if hide_tick_labels:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect('equal')
    if data is not None:
        plt.scatter(data[:, 0], data[:, 1], s=16)
    plt.show()


n_grid = 5
codebook = generate_codebook(n_grid)
plot_data_grid_with_resids(data, n_grid=n_grid, show_next_level_grid=True, show_centroids=True,
                           hide_tick_labels=False, codebook=codebook, show_resids=False)
plot_data_grid_with_resids(data, n_grid=n_grid, show_next_level_grid=True, show_centroids=True,
                           hide_tick_labels=True, codebook=codebook, show_resids=True)
plot_data_grid_with_resids(data, n_grid=n_grid, show_next_level_grid=False, show_centroids=False,
                           hide_tick_labels=False, codebook=np.zeros((data.shape[0],2)), show_resids=True, show_grid=False)
