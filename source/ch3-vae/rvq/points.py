import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def quantizer(data, codebooks, n_grid=5):
    "this will spit out indices for residuals in a series of 'nested' codebooks"
    resids = data
    indices = []
    for cb in codebooks:
        indices_l = get_region_membership(resids, codebook=cb)
        resids = resids - cb[indices_l]
        indices.append(indices_l)
    return np.array(indices)
def generate_codebook(n_grid, n_dim=2, debug=False):
    n_gridpoints = n_grid ** n_dim
    if debug: print(f"generate_codebook: n_grid = {n_grid}, n_dim = {n_dim}, n_gridpoints = {n_gridpoints}")
    centroids = np.empty((n_gridpoints, n_dim))
    h = (DATA_MAX - DATA_MIN)/n_grid
    for i in range(n_dim):
        coords = np.linspace(DATA_MIN + h/2, DATA_MAX - h/2, n_grid)
        coords = np.tile(coords, int(n_gridpoints / n_grid ** (i + 1)))
        coords = np.repeat(coords, n_grid ** i)
        centroids[:, i] = coords
    return centroids

def distance(p1, p2):
    return np.sum((p1-p2)**2, axis=1)

def calc_cluster_membership(data, centroids):
    npoints = data.shape[0]
    min_ds = 9999*np.ones(npoints)
    cluster_memb = np.zeros(npoints, dtype=int)-1
    for i, c in enumerate(centroids): # compute distances for all points
        ds = distance(data, c)
        inds = np.argwhere(ds < min_ds)
        if inds.size > 0:
            min_ds[inds] = ds[inds]
            cluster_memb[inds] = i
    assert len(cluster_memb)==npoints # we're not including the centroids themselves here
    return cluster_memb

def get_region_membership(data: np.array, h=0.2, codebook=None):
    "Tells which region each point is in. TBD: this is slow but it works! ;-) "
    memb = np.zeros(data.shape[0], dtype=np.int32)
    if codebook is None:  # just assume basic squares
        for di, p in enumerate(data):
            i = (p[0] - DATA_MIN) // h
            j = (p[1] - DATA_MIN) // h
            ind = i + j * n_grid
            memb[di] = ind
    else:
        memb = calc_cluster_membership(data, codebook)
    return memb

# make some data
n_grid = 5
n_points = 25
DATA_MIN, DATA_MAX = -0.5, 0.5 # we'll let these be globals
def plot_random_data():
    np.random.seed(9)  # for reproducibility
    data = DATA_MIN + (DATA_MAX-DATA_MIN)*np.random.rand(n_points, 2)
    # plot it
    fig, ax = plt.subplots(figsize=(3,3))
    ax.set_xlim(DATA_MIN, DATA_MAX)
    ax.set_ylim(DATA_MIN, DATA_MAX)
    #ax.set_xticks([])  # hide axis ticks
    #ax.set_yticks([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.scatter(data[:, 0], data[:, 1], s=16)
    plt.savefig('images/data2d.png')
    plt.show()


if __name__ == '__main__':
    plot_random_data()
