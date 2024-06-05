import torch
import torch.optim as optim
from vqvae import VQVAE, train_vqvae
from data import train_loader
from gen_image import generate_images
import matplotlib.pyplot as plt
from pixelcnn import PixelCNN, train_pixelcnn
import os

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize VQVAE model
    vqvae = VQVAE(latent_dim=16, num_embeddings=128).to(device)

    # Load VQVAE model if exists
    vqvae_path = 'vq_vae.pth'
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path))
        print(f"Loaded VQVAE model from {vqvae_path}")
    else:
        vqvae_optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
        train_vqvae(vqvae, train_loader, vqvae_optimizer, device, epochs=10)
        torch.save(vqvae.state_dict(), vqvae_path)

    # Initialize PixelCNN model
    pixelcnn = PixelCNN(num_embeddings=128).to(device)

    # Load PixelCNN model if exists
    pixelcnn_path = 'pixel_cnn.pth'
    if os.path.exists(pixelcnn_path):
        pixelcnn.load_state_dict(torch.load(pixelcnn_path))
        print(f"Loaded PixelCNN model from {pixelcnn_path}")
    else:
        pixelcnn_optimizer = optim.Adam(pixelcnn.parameters(), lr=1e-3)
        pixelcnn = train_pixelcnn(pixelcnn, train_loader, vqvae, pixelcnn_optimizer, device, epochs=30)
        torch.save(pixelcnn.state_dict(), pixelcnn_path)

    # Generate images
    generated_images = generate_images(pixelcnn, vqvae, device)
    for i in range(10):
        plt.imshow(generated_images[i][0], cmap='gray')
        plt.title("Generated Image")
        plt.show()


