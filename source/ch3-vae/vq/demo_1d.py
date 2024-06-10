import argparse
import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face
from sklearn.preprocessing import KBinsDiscretizer
from skimage import data
from sklearn.metrics import mean_squared_error
from math import log10, sqrt
from scipy.spatial import Voronoi, voronoi_plot_2d

plt.style.use("ggplot")

def load_raccoon_image(gray=True):
    global face
    try:
        raccoon_face = face(gray=gray)
    except ImportError:
        from scipy.misc import face
        raccoon_face = face(gray=gray)
    return raccoon_face

def load_image(image_name, gray=True):
    if image_name == "raccoon":
        return load_raccoon_image(gray)
    elif image_name == "astronaut":
        return data.astronaut() if not gray else data.astronaut()[:, :, 0]
    elif image_name == "camera":
        x = data.camera()
        return x
    else:
        raise ValueError("Unsupported image name. Choose from 'raccoon', 'astronaut', or 'camera'.")

def plot_image_and_histogram(image, title, save_path):
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3.5))
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis("off")
    ax[0].set_title("Rendering of the image", fontsize=8)
    r = image.ravel()
    ax[1].hist(r, bins=256)
    ax[1].set_xlabel("Pixel value", fontsize=8)
    ax[1].set_ylabel("Count of pixels", fontsize=8)
    ax[1].set_title("Distribution of the pixel values", fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def compress_image(image, n_bins=8, strategy='uniform'):
    encoder = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy,
        random_state=0,
    )
    tmp = image.reshape(-1, 1)
    tmp2 = encoder.fit_transform(tmp)
    compressed_image = tmp2.reshape(image.shape)
    bin_edges = encoder.bin_edges_[0]
    bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
    return compressed_image, bin_center

def plot_histograms_with_centers(image, bin_centers_uniform, bin_centers_kmeans, save_path):
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3.5))
    ax[0].hist(image.ravel(), bins=256)
    color = "tab:orange"
    for center in bin_centers_uniform:
        ax[0].axvline(center, color=color)
        ax[0].text(center - 10, ax[0].get_ybound()[1] + 100, f"{center:.1f}", color=color, fontsize=8)
    ax[0].set_xlabel("Pixel value - Uniform Binning", fontsize=8)
    ax[0].set_ylabel("Count of pixels", fontsize=8)

    ax[1].hist(image.ravel(), bins=256)
    for center in bin_centers_kmeans:
        ax[1].axvline(center, color=color)
        ax[1].text(center - 10, ax[1].get_ybound()[1] + 100, f"{center:.1f}", color=color, fontsize=8)
    ax[1].set_xlabel("Pixel value - K-means Binning", fontsize=8)
    ax[1].set_ylabel("Count of pixels", fontsize=8)

    fig.suptitle("Histogram with Bin Centers: Uniform vs K-means", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def calculate_psnr(original, compressed):
    mse = mean_squared_error(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def plot_voronoi(centroids, title, save_path):
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
    plt.title(title)

    # Save plot
    plt.savefig(save_path)
    plt.show()

def main(image_name, n_bins):
    image = load_image(image_name)
    print(f"The dimension of the image is {image.shape}")
    print(f"The data used to encode the image is of type {image.dtype}")
    print(f"The number of bytes taken in RAM is {image.nbytes}")

    plot_image_and_histogram(image, f"Original image of {image_name}", f"{image_name}_original.png")

    compressed_image_uniform, bin_centers_uniform = compress_image(image, n_bins, "uniform")
    compressed_image_kmeans, bin_centers_kmeans = compress_image(image, n_bins, "kmeans")

    plot_image_and_histogram(compressed_image_uniform, f"{image_name} compressed using 3 bits and a uniform strategy",
                             f"{image_name}_compressed_uniform.png")
    plot_image_and_histogram(compressed_image_kmeans, f"{image_name} compressed using 3 bits and a K-means strategy",
                             f"{image_name}_compressed_kmeans.png")

    plot_histograms_with_centers(image, bin_centers_uniform, bin_centers_kmeans, f"{image_name}_dist_combined.png")

    psnr_uniform = calculate_psnr(image, compressed_image_uniform)
    psnr_kmeans = calculate_psnr(image, compressed_image_kmeans)

    print(f"PSNR (uniform): {psnr_uniform}")
    print(f"PSNR (kmeans): {psnr_kmeans}")

    print(f"The number of bytes taken in RAM (uniform): {compressed_image_uniform.nbytes}")
    print(f"Compression ratio (uniform): {compressed_image_uniform.nbytes / image.nbytes}")
    print(f"Type of the compressed image (uniform): {compressed_image_uniform.dtype}")

    print(f"The number of bytes taken in RAM (kmeans): {compressed_image_kmeans.nbytes}")
    print(f"Compression ratio (kmeans): {compressed_image_kmeans.nbytes / image.nbytes}")
    print(f"Type of the compressed image (kmeans): {compressed_image_kmeans.dtype}")

    # Plot Voronoi diagrams for k-means centroids
    plot_voronoi(bin_centers_kmeans.reshape(-1, 1), f"Voronoi Diagram for K-means Binning of {image_name}",
                 f"{image_name}_voronoi_kmeans.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and compress an image.")
    parser.add_argument("--image", type=str, default="raccoon", choices=["raccoon", "astronaut", "camera"],
                        help="Image to process")
    parser.add_argument("--n_bins", type=int, default=8, help="Number of bins for discretization")
    args = parser.parse_args()

    main(args.image, args.n_bins)
