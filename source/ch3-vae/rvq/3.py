from points import *


def plot_codebook2d(codebook):
    x, y = codebook[:, 0], codebook[:, 1]
    plt.figure(figsize=(5, 5))  # Adjust figure size as needed
    plt.scatter(x, y, c='blue', marker='o')  # Scatter plot
    for i, txt in enumerate(range(codebook.shape[0])):
        plt.annotate(txt, (x[i], y[i]), textcoords="offset points", xytext=(0, -12), ha='center', fontsize='small')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('A 2D Codebook', fontsize=10)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig('images/codebook2d_vis.png')
    plt.show()


def find_closest_centroids(data, codebook):
    distances = np.sum((data[:, np.newaxis] - codebook) ** 2, axis=2)
    closest_centroids = np.argmin(distances, axis=1)
    return closest_centroids  # indices not coordinates


def compute_error(data, codebook):
    n_grid = int(np.sqrt(codebook.shape[0]))
    h = 1 / n_grid * (DATA_MAX - DATA_MIN)
    quantized_ind = find_closest_centroids(data, codebook)
    quantized_xy = codebook[quantized_ind]
    error = np.sqrt(np.sum((data - quantized_xy) ** 2))
    return error


def plot_ngrid_error():
    n_points2 = 100
    data2 = DATA_MIN + (DATA_MAX - DATA_MIN) * np.random.rand(n_points2, 2)

    errors = []
    grids = np.array([5, 10, 25, 100, 200])
    for n_grid2 in grids:
        codebook = generate_codebook(n_grid2)
        error = compute_error(data2, codebook)
        errors.append(error)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(grids, errors, 'o-')
    ax[1].loglog(grids, errors, 'o-')
    for i in range(2):
        ax[i].set_ylabel('Error')
        ax[i].set_xlabel('n_grid')
    plt.suptitle('codebook size vs quantization error', fontsize=10)
    plt.tight_layout()
    plt.savefig('images/booksize_vs_error.png')
    plt.show()
    print(f"lowest error (for n_grid={grids[-1]}) = ", errors[-1])


if __name__ == '__main__':
    codebook = generate_codebook(n_grid)
    plot_codebook2d(codebook)
    plot_ngrid_error()
