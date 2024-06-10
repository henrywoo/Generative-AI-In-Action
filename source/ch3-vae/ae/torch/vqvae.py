import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
from pixelcnn import get_pixelcnn  # Assuming you've ported the PixelCNN module
from vqvae_plot import *  # For plotting (ensure these functions work with PyTorch tensors)


LATENT_DIM = 16
NUM_EMBEDDINGS = 128
BATCH_SIZE = 128


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embeddings.weight)  # Initialize embeddings uniformly

    def forward(self, z):
        # Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Compute L2 distance between z and embedding vectors
        distances = (
            z_flattened.pow(2).sum(1, keepdim=True)
            - 2 * z_flattened @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(0, keepdim=True)
        )

        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize and unflatten
        quantized = self.embeddings(encoding_indices).view(z.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Losses
        commitment_loss = F.mse_loss(z.detach(), quantized)
        codebook_loss = F.mse_loss(quantized.detach(), z)
        loss = self.beta * commitment_loss + codebook_loss

        # Straight-Through Estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, latent_dim, num_embeddings):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
        )
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss

from torch.optim import Adam
def train_vqvae(model, train_loader, epochs, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in train_loader:  # Unpack labels if needed
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, vq_loss = model(x)
            recon_loss = F.mse_loss(x_recon, x) / data_variance
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

def load_or_train_vqvae(x_train_scaled, x_test_scaled):
    model_path = Path("mbin/vqvae")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "vqvae_model.pt"

    vqvae = VQVAE(LATENT_DIM, NUM_EMBEDDINGS)

    if model_file.exists():
        vqvae.load_state_dict(torch.load(model_file))
        print("Loaded pre-trained VQ-VAE model.")
    else:
        train_dataset = TensorDataset(torch.tensor(x_train_scaled).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_vqvae(vqvae, train_loader, epochs=30)
        torch.save(vqvae.state_dict(), model_file)

    return vqvae


# Load data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()  # Adjust data loading if needed
x_train = np.expand_dims(x_train, 1)
x_test = np.expand_dims(x_test, 1)
x_train_scaled = (x_train / 255.0) - 0.5
x_test_scaled = (x_test / 255.0) - 0.5

# Load or train VQ-VAE
vqvae = load_or_train_vqvae(x_train_scaled, x_test_scaled)
vqvae.eval()

# ... (Code for generating latent codes, training PixelCNN, and sampling new images is similar but adjusted for PyTorch tensors and model interfaces)
