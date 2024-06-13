import torch
import torch.nn as nn

"""
Step 1: Define the Generator Model Class
Ensure that the Generator class is defined in your inference script exactly as it was when you trained the model.
This class structure must be the same to correctly load the model parameters.
"""
class Generator(nn.Module):
    def __init__(self, latent_dim=64, img_size=(28, 28), channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
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
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.channels, self.img_size[0], self.img_size[1])
        return out

class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        activation = nn.LeakyReLU()
        layers_dim = [self.img_size[0] * self.img_size[1] * self.channels, 512, 256, 128, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.LayerNorm(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        out = x.reshape(-1, self.img_size[0] * self.img_size[1] * self.channels)
        for layer in self.layers:
            out = layer(out)
        return out

if __name__ == "__main__":
    from hiq.vis import print_model
    print_model(Generator())