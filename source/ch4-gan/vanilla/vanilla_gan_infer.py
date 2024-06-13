import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from hiq import set_seed
from models import Generator


"""
Step 1: Initialize the Generator and Load the Checkpoint
You need to create an instance of the Generator class and load the checkpoint into this instance.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator
generator = Generator().to(device)

# Load the weights from the checkpoint
generator.load_state_dict(torch.load('output/epoch_500/generator_ckpt.pth', map_location=device))
generator.eval()  # Set the model to inference mode


"""
Step 2: Generate Images
Now that the model is loaded, you can generate images by passing random noise vectors to the generator.
"""
def generate_images(generator, num_images=64, latent_dim=64):
    with torch.no_grad():
        # Generate random latent vectors
        z = torch.randn(num_images, latent_dim, device=device)
        # Generate images from the latent vectors
        fake_images = generator(z)
        fake_images = (fake_images + 1) / 2  # Rescale images from [-1, 1] to [0, 1]
        return fake_images

def show_images(images, nrow=8):
    grid = make_grid(images, nrow=nrow)
    plt.figure(figsize=(3, 3))
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.savefig('output/image_500.png')
    plt.show()


if __name__ == '__main__':
    set_seed(has_torch=True)
    fake_images = generate_images(generator, num_images=64, latent_dim=64)
    show_images(fake_images)
