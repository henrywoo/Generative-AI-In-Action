import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data, img_as_float
from skimage.color import rgba2rgb
from voronoi import plot_voronoi
from sklearn.metrics import mean_squared_error
from math import log10, sqrt
import imageio
from urllib.parse import urlparse
import os

plt.style.use('ggplot')

def load_image(image_name):
    if image_name == "astronaut":
        return img_as_float(data.astronaut())
    elif image_name == "camera":
        return img_as_float(data.camera())
    elif image_name == "coins":
        return img_as_float(data.coins())
    elif image_name == "horse":
        return img_as_float(data.horse())
    elif image_name == "rocket":
        return img_as_float(data.rocket())
    elif os.path.isfile(image_name) or urlparse(image_name).scheme in ['http', 'https']:
        image = img_as_float(imageio.imread(image_name))
        if image.shape[-1] == 4:  # Convert RGBA to RGB
            image = rgba2rgb(image)
        return image
    else:
        raise ValueError("Unsupported image name. Choose from predefined images or provide a valid file path/URL.")

def compress_image_kmeans(image, n_colors):
    w, h = image.shape[:2]
    d = image.shape[2] if image.ndim == 3 else 1
    image_array = np.reshape(image, (w * h, d))
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)
    compressed_image = kmeans.cluster_centers_[labels].reshape(w, h, d)
    if d == 1:
        compressed_image = compressed_image[:, :, 0]
    return compressed_image, kmeans.cluster_centers_

def plot_image(image, title, save_path):
    if save_path.startswith("http"):
        save_path = save_path.split("/")[-1]
        title = title.split("/")[-1]
    plt.figure(figsize=(8, 8))
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()

def calculate_psnr(original, compressed):
    original = original.reshape(-1, original.shape[-1] if original.ndim == 3 else 1)
    compressed = compressed.reshape(-1, compressed.shape[-1] if compressed.ndim == 3 else 1)
    mse = mean_squared_error(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Since img_as_float scales the image between 0 and 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main(image_name, n_colors):
    image = load_image(image_name)
    print(f"The dimension of the image is {image.shape}")
    print(f"The data used to encode the image is of type {image.dtype}")
    plot_image(image, f"Original image of {image_name}", f"{image_name}_original.png")
    compressed_image, centroids = compress_image_kmeans(image, n_colors)
    plot_image(compressed_image, f"Compressed image of {image_name} with {n_colors} colors", f"{image_name}_compressed_{n_colors}.png")
    psnr = calculate_psnr(image, compressed_image)
    print(f"PSNR: {psnr}")
    plot_voronoi(centroids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Vector Quantization using k-means clustering.")
    # , choices=["astronaut", "camera", "coins", "horse", "rocket"]
    # --image https://media-cldnry.s-nbcnews.com/image/upload/mpx/2704722219/2024_04/1713444872219_tdy_pop_8a_taylor_swift_book_240418_1920x1080-t6kjrp.jpg
    parser.add_argument("--image", type=str, default="astronaut", help="Image to process")
    parser.add_argument("--n_colors", type=int, default=16, help="Number of colors for quantization")
    args = parser.parse_args()

    main(args.image, args.n_colors)
