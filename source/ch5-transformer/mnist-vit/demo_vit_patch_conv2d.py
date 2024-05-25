import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class PatchExtractor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels=1, out_channels=patch_size**2,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.conv(x)

def load_image(image_path, size, grayscale=True):
    """Loads and preprocesses an image.

    Args:
        image_path (str): Path to the image file.
        size (tuple): Desired output image size.
        grayscale (bool): Whether to convert to grayscale. Defaults to True.
    """
    transforms_list = [transforms.Resize(size)]
    if grayscale:
        transforms_list.append(transforms.Grayscale(num_output_channels=1))  # Ensure single channel
    transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def display_patches(patches, n_cols, output_filename="patches.jpg"):
    # Calculate number of rows
    n_rows = (patches.shape[0] + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i in range(patches.shape[0]):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(patches[i], cmap='gray')
        plt.axis('off')
    plt.savefig(output_filename)
    plt.show()

def main(image_path = 'model-small.jpg'):
    # Parameters (now easier to modify in one place)
    patch_size = 50
    image_size = (400, 400)
    n_cols = 8

    # Load image
    image = load_image(image_path, image_size)

    # Initialize patch extractor
    patch_extractor = PatchExtractor(patch_size=patch_size)

    # Extract patches
    patches = patch_extractor(image)

    # Reshape and prepare for display
    patches = patches.data.squeeze(0).permute(1, 2, 0).contiguous().view(-1, patch_size, patch_size).numpy()

    # Display patches
    display_patches(patches, n_cols, output_filename="model-small-patches-conv2d.jpg")


if __name__ == "__main__":
    main()
