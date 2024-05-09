import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Sample 2D vectors, assume this is already converted from a PyTorch tensor to a NumPy array
x = np.array([
    [ 8.1833e-02, 4.6128e-01, -3.7352e-01, -2.4597e-01, -7.7958e-01, -3.1526e-01, 7.5574e-01, 9.8554e-01],
    [-2.5370e+01, -9.6212e+00, -4.8634e+01, 4.0335e+01, 2.5986e+01, -6.2208e+00, -2.3431e+01, 3.7373e+01],
    [-2.2888e+01, 2.4283e-01, -1.1050e+01, 2.4527e+01, -3.0434e+01, 3.0759e+01, 1.4995e+01, -8.3716e+00],
    [-2.6138e+01, 3.8631e+01, 1.1060e+01, 1.0744e+01, -7.0278e+00, 2.8009e+01, -6.4780e+01, -1.1298e+01],
    [-2.7695e+01, -6.1792e+01, 9.8147e+00, 4.4641e+01, 3.5382e+01, -5.0159e+01, 1.2015e+01, -3.3734e+01],
    [ 1.0575e+01, 2.9678e+01, 4.9503e+00, -3.5092e+01, 2.0950e+01, -2.0808e+01, -3.3910e+01, 1.6339e+01],
    [ 4.6654e-01, 1.7568e+00, -1.9397e+00, -5.0388e-01, 1.5202e+00, -2.4894e+00, 2.0318e+00, -1.6704e+00],
    [ 9.2437e+01, -1.7299e+02, 1.6489e+01, 1.2231e+01, -3.7330e+01, 5.3938e+01, -8.4591e+01, 8.0884e+00],
    [-7.5250e+00, -3.7608e+00, -1.3731e+00, -1.3177e+01, 5.1248e+00, 8.6944e+00, 3.0409e+00, -3.0931e+00],
    [-3.0767e+01, -1.3125e+01, 6.2904e+01, -1.9400e+01, -1.0472e+01, -1.1854e+01, 1.2911e+01, 2.1330e+01]
])

def draw_tensor(x):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE


    # Convert your data into 2D points using t-SNE with adjusted perplexity
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)  # Adjusting perplexity
    x_2d = tsne.fit_transform(x)

    # Plotting the results of t-SNE
    plt.figure(figsize=(8, 6))
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=np.arange(10), cmap='viridis', edgecolor='k', s=50)
    for i in range(10):
        plt.text(x_2d[i, 0], x_2d[i, 1], str(i), color='black', fontweight='bold')

    plt.colorbar(label='Digit label')
    plt.title('t-SNE visualization of digit embeddings')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.grid(True)
    plt.show()
