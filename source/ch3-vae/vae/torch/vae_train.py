import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torchvision import datasets, transforms

# --- Data Handling ---
DATA_PATH = Path() / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load FashionMNIST (using torchvision for convenience)
transform = transforms.Compose([transforms.ToTensor()])  # Convert to PyTorch Tensors
train_dataset = datasets.FashionMNIST(DATA_PATH, train=True, download=True, transform=transform)
valid_dataset = datasets.FashionMNIST(DATA_PATH, train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

# --- Model Definition ---
class VariationalEncoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc_mean = nn.Linear(100, codings_size)
        self.fc_logvar = nn.Linear(100, codings_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mean, log_var

class VariationalDecoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.fc1 = nn.Linear(codings_size, 100)
        self.fc2 = nn.Linear(100, 150)
        self.fc3 = nn.Linear(150, 28 * 28)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z.view(-1, 28, 28)  # Reshape to image format

class VariationalAutoencoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.encoder = VariationalEncoder(codings_size)
        self.decoder = VariationalDecoder(codings_size)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var


# --- Loss Function and Training ---
def vae_loss(reconstruction, original, mean, log_var):
    reconstruction_loss = F.mse_loss(reconstruction, original)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# --- Training ---
torch.manual_seed(42)
codings_size = 10

model = VariationalAutoencoder(codings_size)
optimizer = optim.Adam(model.parameters())
epochs = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, log_var = model(data)
        loss = vae_loss(recon_batch, data, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# --- Plotting Reconstructions ---
def plot_reconstructions(model, images, n_images=5):
    model.eval()
    with torch.no_grad():
        reconstructions, _, _ = model(images.to(device))
    reconstructions = reconstructions.cpu().numpy()  # Move back to CPU

    fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(n_images * 1.5, 3))
    for i in range(n_images):
        axes[0, i].imshow(images[i].squeeze(), cmap='binary')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap='binary')
        axes[1, i].axis('off')
    plt.show()

# Get a batch of images from the validation set for plotting
sample_data, _ = next(iter(valid_loader))
plot_reconstructions(model, sample_data[:5])

# --- Save the Model ---
MODEL_PATH = Path() / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH / "vae_fashion_mnist.pth")