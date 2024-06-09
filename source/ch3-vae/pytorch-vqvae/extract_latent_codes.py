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
            latents = vqvae_model.encode(data)  # Directly using encode method to get latents
            latents = latents.argmax(dim=1)  # Get the indices of the closest embeddings
            all_latents.append(latents.cpu())

    all_latents = torch.cat(all_latents)
    return all_latents

# Example usage to extract latent codes and save them
def save_latent_codes():
    batch_size = 32
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae_model = VectorQuantizedVAE(input_dim=1, dim=256, K=512)  # Adjust parameters based on your model definition
    vqvae_model = load_model(vqvae_model, file_vqvae, device)

    latent_codes = extract_latent_codes(vqvae_model, dataloader, device)

    # Save the latent codes to a file
    torch.save(latent_codes, file_codebook)

save_latent_codes()


