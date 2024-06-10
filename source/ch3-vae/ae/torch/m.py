import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
from vqvae_plot import plot_original_vs_code, show_all_subplots  # Make sure this module is available

LATENT_DIM = 16
NUM_EMBEDDINGS = 64
BATCH_SIZE = 128


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


# Define the VQVAE model
class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, num_embeddings=64):
        super(VQVAE, self).__init__()
        self.encoder = self.build_encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = self.build_decoder(latent_dim)

    def build_encoder(self, latent_dim):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1)
        )

    def build_decoder(self, latent_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, encoding_indices = self.quantizer(encoded)
        reconstructions = self.decoder(quantized)
        return reconstructions, vq_loss, encoding_indices


# Train the VQVAE model
def train_vqvae(model, train_loader, num_epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            reconstructions, vq_loss, _ = model(data)
            recon_loss = F.mse_loss(reconstructions, data)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader)}')


# Function to load or train VQVAE
def load_or_train_vqvae():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_path = Path("mbin/vqvae")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "vqvae_model.pth"
    vqvae_model = VQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)

    if model_file.exists():
        vqvae_model.load_state_dict(torch.load(model_file))
    else:
        train_vqvae(vqvae_model, train_loader, num_epochs=30)
        torch.save(vqvae_model.state_dict(), model_file)

    return vqvae_model, train_loader, test_loader


# Use the function to load or train VQVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae_model, train_loader, test_loader = load_or_train_vqvae()

# Test the VQVAE model
vqvae_model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        reconstructions, _, _ = vqvae_model(data)
        data = data.cpu().numpy()
        reconstructions = reconstructions.cpu().numpy()
        show_all_subplots(data[:10], reconstructions[:10])
        break

# Generate the codebook indices using randomly selected test images
test_images = []
encoded_outputs = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        encoded = vqvae_model.encoder(data)
        test_images.append(data.cpu().numpy())
        encoded_outputs.append(encoded.cpu().numpy())
        break  # Use only the first batch for plotting

test_images = np.concatenate(test_images, axis=0)
encoded_outputs = np.concatenate(encoded_outputs, axis=0)

# Select random test images
idx = np.random.choice(len(test_images), 10, replace=False)
selected_test_images = test_images[idx]

# Flatten the encoded outputs correctly
batch_size, channels, height, width = encoded_outputs.shape
flat_enc_outputs = torch.tensor(encoded_outputs).view(-1, channels).to(device)

# Get codebook indices
codebook_indices = vqvae_model.quantizer.get_code_indices(flat_enc_outputs).cpu().numpy()

# Reshape codebook indices correctly
codebook_indices = codebook_indices.reshape(batch_size, height, width)

# Select corresponding code indices for the selected test images
selected_codebook_indices = codebook_indices[idx]

# Plot original vs codebook indices using selected test images
plot_original_vs_code(selected_test_images, selected_codebook_indices)
