from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
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
        input_shape = x.shape
        flattened = x.view(-1, self.embedding_dim)
        encoding_indices = self.get_code_indices(flattened)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        quantized = torch.matmul(encodings, self.embeddings.t())
        quantized = quantized.view(*input_shape)
        commitment_loss = torch.mean((quantized.detach() - x) ** 2)
        codebook_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = commitment_loss * self.beta + codebook_loss
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


class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, num_embeddings=64):
        super(VQVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.encoder = self.build_encoder()
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = self.build_decoder()

    def build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.latent_dim, kernel_size=1)
        )

    def build_decoder(self):
        return nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss


class VQVAETrainer:
    def __init__(self, train_variance, latent_dim=16, num_embeddings=64):
        self.train_variance = train_variance
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.model = VQVAE(latent_dim, num_embeddings)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "vqvae_loss": []
        }

    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()
        reconstructions, vq_loss = self.model(x)
        reconstruction_loss = torch.mean((x - reconstructions) ** 2) / self.train_variance
        total_loss = reconstruction_loss + vq_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), reconstruction_loss.item(), vq_loss.item()

    def fit(self, x, epochs=1, batch_size=32):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            reconstruction_loss = 0
            vqvae_loss = 0
            for batch in dataloader:
                batch_x, = batch
                losses = self.train_step(batch_x)
                total_loss += losses[0]
                reconstruction_loss += losses[1]
                vqvae_loss += losses[2]
            self.loss_history["total_loss"].append(total_loss / len(dataloader))
            self.loss_history["reconstruction_loss"].append(reconstruction_loss / len(dataloader))
            self.loss_history["vqvae_loss"].append(vqvae_loss / len(dataloader))
            print(f"Total Loss: {total_loss / len(dataloader)}, "
                  f"Reconstruction Loss: {reconstruction_loss / len(dataloader)}, "
                  f"VQ-VAE Loss: {vqvae_loss / len(dataloader)}")


def create_vqvae_model(latent_dim=16, num_embeddings=64):
    return VQVAE(latent_dim=latent_dim, num_embeddings=num_embeddings)


def load_or_train_vqvae():
    model_path = Path("mbin/vqvae")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "vqvae_model.pth"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    x_train_scaled = np.concatenate([batch[0].numpy() for batch in train_loader], axis=0)
    x_test_scaled = np.concatenate([batch[0].numpy() for batch in test_loader], axis=0)
    data_variance = np.var(x_train_scaled)
    vqvae_trainer = VQVAETrainer(data_variance, latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS)
    if model_file.exists():
        vqvae_trainer.model.load_state_dict(torch.load(model_file))
    else:
        vqvae_trainer.fit(x_train_scaled, epochs=30, batch_size=BATCH_SIZE)
        plot_training_losses(vqvae_trainer.loss_history)
        torch.save(vqvae_trainer.model.state_dict(), model_file)
    return vqvae_trainer, x_train_scaled, x_test_scaled


vqvae_trainer, x_train_scaled, x_test_scaled = load_or_train_vqvae()
trained_vqvae_model = vqvae_trainer.model
print(trained_vqvae_model)

idx = np.random.choice(len(x_test_scaled), 10)
test_images = torch.tensor(x_test_scaled[idx], dtype=torch.float32)
with torch.no_grad():
    reconstructions_test, _ = trained_vqvae_model(test_images)
show_all_subplots(test_images.numpy(), reconstructions_test.numpy())

encoder = vqvae_trainer.model.encoder
quantizer = vqvae_trainer.model.quantizer
with torch.no_grad():
    encoded_outputs = encoder(test_images)
flat_enc_outputs = encoded_outputs.view(-1, encoded_outputs.shape[-1])
codebook_indices = quantizer.get_code_indices(flat_enc_outputs)
codebook_indices = codebook_indices.view(encoded_outputs.shape[:-1])
plot_original_vs_code(test_images.numpy(), codebook_indices.numpy())
