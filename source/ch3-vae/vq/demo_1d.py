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
        return data.camera()
    else:
        raise ValueError("Unsupported image name. Choose from 'raccoon', 'astronaut', or 'camera'.")


def plot_image_and_histogram(image, title, save_path):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].axis("off")
    ax[0].set_title("Rendering of the image")
    r = image.ravel()
    ax[1].hist(r, bins=256)
    ax[1].set_xlabel("Pixel value")
    ax[1].set_ylabel("Count of pixels")
    ax[1].set_title("Distribution of the pixel values")
    fig.suptitle(title)
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
    fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
    ax[0].hist(image.ravel(), bins=256)
    color = "tab:orange"
    for center in bin_centers_uniform:
        ax[0].axvline(center, color=color)
        ax[0].text(center - 10, ax[0].get_ybound()[1] + 100, f"{center:.1f}", color=color)
    ax[0].set_xlabel("Pixel value - Uniform Binning")
    ax[0].set_ylabel("Count of pixels")

    ax[1].hist(image.ravel(), bins=256)
    for center in bin_centers_kmeans:
        ax[1].axvline(center, color=color)
        ax[1].text(center - 10, ax[1].get_ybound()[1] + 100, f"{center:.1f}", color=color)
    ax[1].set_xlabel("Pixel value - K-means Binning")
    ax[1].set_ylabel("Count of pixels")

    fig.suptitle("Histogram with Bin Centers: Uniform vs K-means")
    plt.savefig(save_path)
    plt.show()


def calculate_psnr(original, compressed):
    mse = mean_squared_error(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and compress an image.")
    parser.add_argument("--image", type=str, default="raccoon", choices=["raccoon", "astronaut", "camera"],
                        help="Image to process")
    parser.add_argument("--n_bins", type=int, default=8, help="Number of bins for discretization")
    args = parser.parse_args()

    main(args.image, args.n_bins)
