import argparse
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle


def load_image(image_name: str):
    """Load a sample image from sklearn's dataset."""
    if image_name not in ["china.jpg", "flower.jpg"]:
        print("Error: Unsupported image. Please use 'china.jpg' or 'flower.jpg'.")
        sys.exit(1)

    if image_name == "china.jpg":
        return load_sample_image("china.jpg")
    else:
        return load_sample_image("flower.jpg")


def preprocess_image(image):
    """Preprocess the image for clustering."""
    image = np.array(image, dtype=np.float64) / 255
    w, h, d = original_shape = tuple(image.shape)
    assert d == 3
    return image, w, h, d, np.reshape(image, (w * h, d))


def fit_kmeans(image_array, n_colors, sample_size=1000):
    """Fit the KMeans model on a subsample of the image data."""
    image_array_sample = shuffle(image_array, random_state=0, n_samples=sample_size)
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    return kmeans


def predict_labels(model, image_array):
    """Predict labels for the image data using the fitted model."""
    return model.predict(image_array)


def predict_random_labels(codebook_random, image_array):
    """Predict labels using random codebook."""
    return pairwise_distances_argmin(codebook_random, image_array, axis=0)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels."""
    return codebook[labels].reshape(w, h, -1)


def save_image(image, title, filename):
    """Save the image with the given title and filename."""
    os.makedirs("images", exist_ok=True)
    filepath = os.path.join("images", filename)
    plt.figure()
    plt.clf()
    plt.axis("off")
    plt.title(title)
    plt.imshow(image)
    plt.savefig(filepath)


def main(image_name, n_colors, sample_size):
    # Load and preprocess the image
    image = load_image(image_name)
    image, w, h, d, image_array = preprocess_image(image)

    # Fit KMeans model
    print("Fitting model on a small sub-sample of the data")
    t0 = time()
    kmeans = fit_kmeans(image_array, n_colors, sample_size)
    print(f"done in {time() - t0:.3f}s.")

    # Predict labels for KMeans
    print("Predicting color indices on the full image (k-means)")
    t0 = time()
    labels = predict_labels(kmeans, image_array)
    print(f"done in {time() - t0:.3f}s.")

    # Predict labels for Random
    print("Predicting color indices on the full image (random)")
    t0 = time()
    codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
    labels_random = predict_random_labels(codebook_random, image_array)
    print(f"done in {time() - t0:.3f}s.")

    # Save all results
    base_name = os.path.splitext(image_name)[0]
    save_image(image, "Original image (96,615 colors)", f"{base_name}_original.png")
    save_image(recreate_image(kmeans.cluster_centers_, labels, w, h),
               f"Quantized image ({n_colors} colors, K-Means)", f"{base_name}_quantized_kmeans.png")
    save_image(recreate_image(codebook_random, labels_random, w, h),
               f"Quantized image ({n_colors} colors, Random)", f"{base_name}_quantized_random.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image quantization using K-Means clustering")
    parser.add_argument("--image", type=str, default="flower.jpg", choices=["china.jpg", "flower.jpg"],
                        help="The name of the image to load ('china.jpg' or 'flower.jpg')")
    parser.add_argument("--n_colors", type=int, default=64, help="The number of colors to quantize the image to")
    parser.add_argument("--sample_size", type=int, default=1000, help="The number of samples for K-Means fitting")

    args = parser.parse_args()
    main(args.image, args.n_colors, args.sample_size)
