import numpy as np
import matplotlib.pyplot as plt

# make some data
n_points = 25
DATA_MIN, DATA_MAX = -0.5, 0.5 # we'll let these be globals
np.random.seed(9)  # for reproducibility
data = DATA_MIN + (DATA_MAX-DATA_MIN)*np.random.rand(n_points, 2)

def plot_data_grid(data, n_grid=5, hide_tick_labels=True, show_indices=False, show_centroids=False,
                   show_next_level_grid=False):
    "big ol' workhorse plotting routine that we'll progressively make use of as the lesson proceeds"
    fig, ax = plt.subplots(figsize=(3, 3))
    h = 1.0 / n_grid
    ax.set_xlim(DATA_MIN, DATA_MAX)
    ax.set_ylim(DATA_MIN, DATA_MAX)

    for i in range(n_grid + 1):
        ax.axhline(DATA_MIN + i * h, color='black')
        ax.axvline(DATA_MIN + i * h, color='black')

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
                ax.plot(x, y, 'ro', markersize=6)

    if hide_tick_labels:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect('equal')
    if data is not None:
        plt.scatter(data[:, 0], data[:, 1], s=16)
    plt.show()


n_grid = 5
# plot_data_grid(data, n_grid=n_grid)
plot_data_grid(data, n_grid=n_grid, show_indices=True)
plot_data_grid(data, n_grid=n_grid, show_indices=False, show_centroids=True)
plot_data_grid(data, n_grid=n_grid, show_indices=True, show_centroids=True, hide_tick_labels=False)


