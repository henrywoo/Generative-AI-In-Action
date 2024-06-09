import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from modules import VectorQuantizedVAE, VQEmbedding, ResBlock, weights_init  # Ensure this imports your VQ-VAE model definition

file_vqvae = 'models/vqvae/best.pt'
file_codebook = 'models/vqvae/latent_codes.pt'

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
            z_e_x = vqvae_model.encoder(data)  # Encoder output shape (batch_size, 256, 7, 7)
            indices = vqvae_model.codebook(z_e_x)  # Indices shape (batch_size, 7, 7)

            # Convert indices to embeddings
            z_q_x = vqvae_model.codebook.embedding(indices).permute(0, 3, 1, 2)  # Shape (batch_size, 256, 7, 7)
            all_latents.append(z_q_x.cpu())

    all_latents = torch.cat(all_latents)
    return all_latents

# Example usage to extract latent codes and save them
def save_latent_codes():
    batch_size = 32
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_model = VectorQuantizedVAE(input_dim=1, dim=256, k=512)  # Adjust parameters based on your model definition
    vqvae_model = load_model(vqvae_model, file_vqvae, device)

    latent_codes = extract_latent_codes(vqvae_model, dataloader, device)
    print(f"Final latent codes shape: {latent_codes.shape}")  # Check the shape of the latent codes
    # Final latent codes shape: torch.Size([60000, 256, 7, 7])

    # Save the latent codes to a file
    torch.save(latent_codes, file_codebook)

save_latent_codes()
