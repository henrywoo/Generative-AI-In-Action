import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


def train_gan(generator, discriminator, dataloader, num_epochs, noise_dim, device):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            real_imgs = data[0].to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Real images
            valid = torch.ones(batch_size, 1).to(device)
            real_loss = criterion(discriminator(real_imgs), valid)

            # Fake images
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_imgs = generator(z)
            fake = torch.zeros(batch_size, 1).to(device)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            valid = torch.ones(batch_size, 1).to(device)
            g_loss = criterion(discriminator(fake_imgs), valid)

            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')


# Hyperparameters
data_dim = 28 * 28
noise_dim = 100
num_epochs = 50
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset and dataloader
x = torch.randn(1000, data_dim)  # Generate some random data
dataset = TensorDataset(x)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(noise_dim, data_dim)
discriminator = Discriminator(data_dim)

# Train GAN
train_gan(generator, discriminator, dataloader, num_epochs, noise_dim, device)
