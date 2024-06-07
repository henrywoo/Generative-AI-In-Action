import argparse
import torch
import os
import matplotlib.pyplot as plt
from model import VariationalAutoencoder
from vae_train import load_data
from pathlib import Path

torch.manual_seed(0)

here = os.path.abspath(os.path.dirname(__file__))

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    model = VariationalAutoencoder(checkpoint['codings_size']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint['codings_size']

def generate_images(model, device, num_images=12, codings_size=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, codings_size).to(device)
        generated_images = model.decoder(z)
        generated_images = generated_images.view(-1, 1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(3, 4, figsize=(5, 3))
    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(generated_images[i].squeeze(), cmap='binary')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(here, 'generated_images.png'))
    plt.show()

def semantic_interpolation(model, device, codings_size=10, num_steps=8):
    model.eval()
    with torch.no_grad():
        z1 = torch.randn(1, codings_size).to(device)
        z2 = torch.randn(1, codings_size).to(device)

        interpolations = [z1 * (1 - alpha) + z2 * alpha for alpha in torch.linspace(0, 1, num_steps)]
        interpolations = torch.cat(interpolations, dim=0)

        interpolated_images = model.decoder(interpolations)
        interpolated_images = interpolated_images.view(-1, 1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 0.5, 1.5))
    for i, ax in enumerate(axes):
        ax.imshow(interpolated_images[i].squeeze(), cmap='binary')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(here, 'semantic_interpolation.png'))
    plt.show()

def plot_latent_space(model, dataloader, device):
    model.eval()
    all_mean = []
    all_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)  # Move data to the correct device
            mean, logvar = model.encoder(data)
            all_mean.append(mean.cpu())  # Move mean to CPU
            all_labels.append(labels.cpu())  # Move labels to CPU

    all_mean = torch.cat(all_mean)
    all_labels = torch.cat(all_labels)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(all_mean[:, 0].numpy(), all_mean[:, 1].numpy(), c=all_labels.numpy(), cmap='tab10', alpha=0.7)
    legend1 = plt.legend(*scatter.legend_elements(), title="Digits")
    plt.gca().add_artist(legend1)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Distribution')
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, codings_size = load_checkpoint(args.checkpoint_path, device)
    generate_images(model, device, num_images=args.num_images, codings_size=codings_size)
    semantic_interpolation(model, device, codings_size=codings_size, num_steps=args.num_steps)

    DATA_PATH = Path(args.data_path)
    train_loader, valid_loader = load_data(DATA_PATH, args.batch_size)
    plot_latent_space(model, valid_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with a trained Variational Autoencoder")
    parser.add_argument('--checkpoint_path', type=str, default=f"{here}/models/vae_fashion_mnist.pth", help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--num_images', type=int, default=12, help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')

    args = parser.parse_args()
    main(args)
