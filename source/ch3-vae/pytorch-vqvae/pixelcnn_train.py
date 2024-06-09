import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from modules import VectorQuantizedVAE
file_vqvae = 'models/vqvae/best.pt'
file_codebook = 'models/vqvae/latent_codes.pt'
file_pixelcnn = 'models/vqvae/best_pixelcnn.pt'

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(PixelCNN, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(128, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Function to load the VQ-VAE model
def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to extract latent codes
def extract_latent_codes(vqvae_model, dataloader, device):
    vqvae_model.eval()
    all_latents = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            z_e_x = vqvae_model.encoder(data)
            latents = vqvae_model.codebook(z_e_x).argmax(dim=1)  # Get the indices of the closest embeddings
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents)
    return all_latents

# Example usage to extract latent codes and save them
def save_latent_codes():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_model = VectorQuantizedVAE(input_dim=1, dim=256, K=512)  # Adjust parameters based on your model definition
    vqvae_model = load_model(vqvae_model, file_vqvae, device)

    latent_codes = extract_latent_codes(vqvae_model, dataloader, device)

    # Save the latent codes to a file
    torch.save(latent_codes, file_codebook)

save_latent_codes()

# Assuming you've already saved the latent codes
latent_codes = torch.load(file_codebook)

# Assuming latent codes have shape (N, 8, 8) for 8x8 spatial dimension
latent_codes = latent_codes.unsqueeze(1)  # Add channel dimension (N, 1, 8, 8)

dataset = TensorDataset(latent_codes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Function to train the PixelCNN model
def train_pixelcnn(pixelcnn_model, dataloader, device, epochs=10, lr=1e-3):
    optimizer = optim.Adam(pixelcnn_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    pixelcnn_model.to(device)

    for epoch in range(epochs):
        pixelcnn_model.train()
        total_loss = 0

        for batch in dataloader:
            latents = batch[0].to(device)  # (batch_size, 1, 8, 8)
            latents = latents.long()  # Ensure latents are long type for cross-entropy loss

            optimizer.zero_grad()
            output = pixelcnn_model(latents.float())  # Convert latents to float for convolutional layers

            # Debugging prints
            print(f"Latents min: {latents.min().item()}, max: {latents.max().item()}")
            print(f"Output shape: {output.shape}, Latents shape: {latents.squeeze(1).shape}")
            print(f"Output min: {output.min().item()}, max: {output.max().item()}")

            loss = criterion(output, latents.squeeze(1))  # Remove channel dimension for target

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# Example usage to train the PixelCNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_embeddings = 512
embed_dim = 256

pixelcnn_model = PixelCNN(num_embeddings=num_embeddings, embed_dim=embed_dim)
train_pixelcnn(pixelcnn_model, dataloader, device, epochs=20, lr=1e-3)

# Save the trained PixelCNN model
torch.save(pixelcnn_model.state_dict(), file_pixelcnn)
