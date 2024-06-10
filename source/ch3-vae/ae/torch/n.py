import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from vqvae_plot import *

LATENT_DIM = 16
NUM_EMBEDDINGS = 128
BATCH_SIZE = 128

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Parameter(torch.randn(embedding_dim, num_embeddings))

    def forward(self, x):
        flattened = x.view(-1, self.embedding_dim)
        encoding_indices = self.get_code_indices(flattened)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type_as(flattened)
        quantized = torch.matmul(encodings, self.embeddings.T)
        quantized = quantized.view_as(x)

        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = self.beta * commitment_loss + codebook_loss

        quantized = x + (quantized - x).detach()
        return quantized, loss

    def get_code_indices(self, flattened_inputs):
        distances = (
            torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=0)
            - 2 * torch.matmul(flattened_inputs, self.embeddings)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, num_embeddings=64):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

class VQVAETrainer:
    def __init__(self, model, data_variance, lr=1e-3):
        self.model = model
        self.data_variance = data_variance
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "vqvae_loss": []
        }

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            total_loss = 0
            reconstruction_loss = 0
            vqvae_loss = 0
            for x, _ in train_loader:
                x = x.float()
                self.optimizer.zero_grad()
                x_recon, vq_loss = self.model(x)
                recon_loss = F.mse_loss(x_recon, x) / self.data_variance
                loss = recon_loss + vq_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                reconstruction_loss += recon_loss.item()
                vqvae_loss += vq_loss.item()

            total_loss /= len(train_loader)
            reconstruction_loss /= len(train_loader)
            vqvae_loss /= len(train_loader)

            self.loss_history["total_loss"].append(total_loss)
            self.loss_history["reconstruction_loss"].append(reconstruction_loss)
            self.loss_history["vqvae_loss"].append(vqvae_loss)

            print(f"Epoch {epoch + 1}, Total Loss: {total_loss}, Reconstruction Loss: {reconstruction_loss}, VQ-VAE Loss: {vqvae_loss}")

def load_or_train_vqvae():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    data_variance = np.var(train_loader.dataset.data.numpy() / 255.0)
    vqvae = VQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    trainer = VQVAETrainer(vqvae, data_variance)

    trainer.train(train_loader, epochs=30)
    plot_training_losses(trainer.loss_history)

    return vqvae, train_loader.dataset.data.numpy(), train_loader.dataset.targets.numpy()

vqvae, x_train, y_train = load_or_train_vqvae()
x_train_scaled = (x_train / 255.0) - 0.5

# Test the model
vqvae.eval()
idx = np.random.choice(len(x_train_scaled), 10)
test_images = torch.tensor(x_train_scaled[idx]).unsqueeze(1).float()
with torch.no_grad():
    reconstructions_test, _ = vqvae(test_images)

show_all_subplots(test_images.numpy(), reconstructions_test.numpy())

# Quantizer visualization
encoded_outputs = vqvae.encoder(test_images).detach().numpy()
flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
codebook_indices = vqvae.quantizer.get_code_indices(torch.tensor(flat_enc_outputs))
codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])
plot_original_vs_code(test_images.numpy(), codebook_indices)
