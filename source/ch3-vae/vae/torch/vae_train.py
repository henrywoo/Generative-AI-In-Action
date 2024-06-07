import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms

# --- Data Handling ---
def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])  # Convert to PyTorch Tensors
    train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    valid_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

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

def train(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, log_var = model(data)
        loss = vae_loss(recon_batch, data, mean, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)

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

def main(args):
    # Paths
    DATA_PATH = Path(args.data_path)
    MODEL_PATH = Path(args.model_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, valid_loader = load_data(DATA_PATH, args.batch_size)

    # Model
    model = VariationalAutoencoder(args.codings_size).to(device)
    optimizer = optim.Adam(model.parameters())
    torch.manual_seed(args.seed)

    # Training
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch: {epoch} Average loss: {train_loss:.4f}')

    # Plotting Reconstructions
    sample_data, _ = next(iter(valid_loader))
    plot_reconstructions(model, sample_data[:5])

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH / "vae_fashion_mnist.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder for FashionMNIST")
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='models', help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--codings_size', type=int, default=10, help='Size of the latent codings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
