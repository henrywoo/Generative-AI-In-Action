import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data, img_as_float
from sklearn.metrics import mean_squared_error
from math import log10, sqrt

def load_image(image_name):
    if image_name == "astronaut":
        return img_as_float(data.astronaut())
    elif image_name == "camera":
        return img_as_float(data.camera())
    else:
        raise ValueError("Unsupported image name. Choose from 'astronaut' or 'camera'.")

def compress_image_kmeans(image, n_colors):
    w, h, d = image.shape
    image_array = np.reshape(image, (w * h, d))

    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)
    labels = kmeans.predict(image_array)

    compressed_image = kmeans.cluster_centers_[labels].reshape(w, h, d)
    return compressed_image, kmeans.cluster_centers_

def plot_image(image, title, save_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()

def calculate_psnr(original, compressed):
    original = original.reshape(-1, original.shape[-1])
    compressed = compressed.reshape(-1, compressed.shape[-1])
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

    compressed_image, _ = compress_image_kmeans(image, n_colors)
    plot_image(compressed_image, f"Compressed image of {image_name} with {n_colors} colors", f"{image_name}_compressed_{n_colors}.png")

    psnr = calculate_psnr(image, compressed_image)
    print(f"PSNR: {psnr}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Vector Quantization using k-means clustering.")
    parser.add_argument("--image", type=str, default="astronaut", choices=["astronaut", "camera"], help="Image to process")
    parser.add_argument("--n_colors", type=int, default=16, help="Number of colors for quantization")
    args = parser.parse_args()

    main(args.image, args.n_colors)
