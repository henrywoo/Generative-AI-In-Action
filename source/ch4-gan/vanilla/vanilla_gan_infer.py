import torch
import torch.nn as nn


"""
Step 1: Define the Generator Model Class
Ensure that the Generator class is defined in your inference script exactly as it was when you trained the model.
This class structure must be the same to correctly load the model parameters.
"""
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 64
        self.img_size = (28, 28)
        self.channels = 1
        activation = nn.LeakyReLU()
        layers_dim = [self.latent_dim, 128, 256, 512, self.img_size[0] * self.img_size[1] * self.channels]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.BatchNorm1d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, z):
        out = z
        for layer in self.layers:
            out = layer(out)
        out = out.view(-1, self.channels, self.img_size[0], self.img_size[1])
        return out

"""
Step 2: Initialize the Generator and Load the Checkpoint
You need to create an instance of the Generator class and load the checkpoint into this instance.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the generator
generator = Generator().to(device)

# Load the weights from the checkpoint
generator.load_state_dict(torch.load('generator_ckpt.pth', map_location=device))
generator.eval()  # Set the model to inference mode


"""
Step 3: Generate Images
Now that the model is loaded, you can generate images by passing random noise vectors to the generator.
"""
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)), interpolation='nearest')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    fake_images = generate_images(generator, num_images=64, latent_dim=64)
    show_images(fake_images)
