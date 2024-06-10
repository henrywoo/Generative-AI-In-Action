import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_voronoi(centroids):
    if centroids.shape[1] < 2:
        print("Voronoi diagram requires at least 2D data. Skipping Voronoi plot for grayscale image.")
        return
    vor = Voronoi(centroids[:, :2])  # Take only the first two dimensions for Voronoi plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5, line_alpha=0.6)
    # Plot the centroids
    ax.plot(centroids[:, 0], centroids[:, 1], 'r*', markersize=10)
    # Set plot limits
    ax.set_xlim(centroids[:, 0].min() - 1, centroids[:, 0].max() + 1)
    ax.set_ylim(centroids[:, 1].min() - 1, centroids[:, 1].max() + 1)
    # Add title
    plt.title("Vector Quantization with Voronoi Diagram")
    # Show plot
    plt.show()


def demo():
    # Generate synthetic 2D data
    np.random.seed(0)
    n_samples = 300
    X = np.random.randn(n_samples, 2)

    # Perform k-means clustering
    n_clusters = 16
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centroids = kmeans.cluster_centers_

    # Create a Voronoi diagram based on the cluster centroids
    vor = Voronoi(centroids)

    # Plot the Voronoi diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5, line_alpha=0.6)

    # Plot the centroids
    ax.plot(centroids[:, 0], centroids[:, 1], 'r*', markersize=10)

    # Set plot limits
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add title
    plt.title("Vector Quantization")

    # Show plot
    plt.show()

if __name__ == '__main__':
    demo()
