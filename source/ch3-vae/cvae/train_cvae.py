import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from hiq.cv_torch import get_cv_dataset, DS_PATH_MNIST

# Parameters
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 100
num_classes = 10


def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    loader_params = dict(
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    dataloader = get_cv_dataset(path=str(data_path),
                                batch_size=batch_size,
                                transform=transform,
                                return_loader=True,
                                convert_rgb=False,
                                **loader_params)
    return dataloader['train'], dataloader['test']


train_loader, test_loader = load_data(DS_PATH_MNIST, batch_size)


# Encoder network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc2_log_var = nn.Linear(intermediate_dim, latent_dim)
        self.fc3 = nn.Linear(num_classes, latent_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(h)
        z_log_var = self.fc2_log_var(h)
        yh = self.fc3(y)
        return z_mean, z_log_var, yh


# Decoder network
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, original_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed


# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x, y):
        z_mean, z_log_var, yh = self.encoder(x, y)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var, yh


# Loss function
def loss_function(x, x_reconstructed, z_mean, z_log_var, yh):
    xent_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.pow(z_mean - yh, 2) - torch.exp(z_log_var))
    return xent_loss + kl_loss


# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.view(-1, original_dim).to(device)
        labels = F.one_hot(labels, num_classes).float().to(device)

        # Check input range
        assert torch.min(data) >= 0.0 and torch.max(data) <= 1.0, "Input data is out of range [0, 1]"

        optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var, yh = vae(data, labels)

        # Check shapes
        assert data.shape == x_reconstructed.shape, f"Shape mismatch: {data.shape} vs {x_reconstructed.shape}"

        loss = loss_function(data, x_reconstructed, z_mean, z_log_var, yh)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset)}')

# Visualization
vae.eval()
with torch.no_grad():
    z_means = []
    labels = []
    for data, label in test_loader:
        data = data.view(-1, original_dim).to(device)
        label = label.to(device)
        z_mean, _, _ = vae.encoder(data, F.one_hot(label, num_classes).float().to(device))
        z_means.append(z_mean)
        labels.append(label)
    z_means = torch.cat(z_means).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

plt.figure(figsize=(6, 6))
plt.scatter(z_means[:, 0], z_means[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.show()

# Generating digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

output_digit = 9  # specify the digit to generate

with torch.no_grad():
    yh = vae.encoder.fc3(torch.eye(num_classes)[output_digit].to(device)).cpu().numpy()
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + yh[0][1]
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + yh[0][0]

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]], device=device).float()
            x_decoded = vae.decoder(z_sample).cpu().numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
