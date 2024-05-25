import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path):
    """Loads an image as a NumPy array."""
    return np.array(Image.open(image_path))

def split_into_patches(image, patch_size):
    """Splits an image into patches.

    Args:
        image (np.ndarray): The input image.
        patch_size (int): Size of the patches (width and height).
    """
    M, N = image.shape[0] // patch_size, image.shape[1] // patch_size
    return image.reshape(M, patch_size, N, patch_size, -1).swapaxes(1,2).reshape(-1, patch_size, patch_size, 3)

def display_patches(patches, n_cols, output_filename="patches.jpg"):
    """Displays extracted patches in a grid and saves the plot.

    Args:
        patches (np.ndarray): Array of extracted patches.
        n_cols (int): Number of columns in the display grid.
        output_filename (str): Filename for saving the plot. Defaults to "patches.jpg".
    """
    n_rows = (len(patches) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    for i, patch in enumerate(patches):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(patch)
        plt.axis('off')
    plt.savefig(output_filename)
    plt.show()

def main(output_filename="model-small-patches-vanilla.jpg"):
    # Parameters
    image_path = 'model-small.jpg'
    patch_size = 50
    n_cols = 8

    # Load and process the image
    image = load_image(image_path)
    patches = split_into_patches(image, patch_size)
    display_patches(patches, n_cols, output_filename)


if __name__ == "__main__":
    main()
