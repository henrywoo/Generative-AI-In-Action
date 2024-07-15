import argparse
import torch
import os
import matplotlib.pyplot as plt
from model import VariationalAutoencoder
from vae_train import load_data, DS_PATH_FASHION_MNIST
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


def plot_histograms(latent_vectors, num_bins=30):
    num_dimensions = latent_vectors.shape[1]
    num_cols = 2
    num_rows = (num_dimensions + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, int(num_rows*1.2)))
    axes = axes.flatten()

    for i in range(num_dimensions):
        ax = axes[i]
        ax.hist(latent_vectors[:, i].numpy(), bins=num_bins, density=True, alpha=0.6, color='g')

        # Plotting the Gaussian distribution for comparison
        mu, std = latent_vectors[:, i].mean(), latent_vectors[:, i].std()
        xmin, xmax = ax.get_xlim()
        x = torch.linspace(xmin, xmax, 100)
        p = torch.exp(-0.5 * ((x - mu) / std) ** 2) / (std * (2 * torch.pi) ** 0.5)
        ax.plot(x.numpy(), p.numpy(), 'k', linewidth=1)
        title = f'Latent Dimension {i + 1}'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

    # Hide any unused subplots
    for i in range(num_dimensions, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(os.path.join(here, 'latent_space_dist.png'))
    plt.show()


def plot_latent_space(model, dataloader, data_name, device):
    model.eval()
    all_mean = []
    all_logvar = []
    all_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)  # Move data to the correct device
            mean, logvar = model.encoder(data)
            all_mean.append(mean.cpu())  # Move mean to CPU
            all_labels.append(labels.cpu())  # Move labels to CPU
            all_logvar.append(logvar.cpu())  # Save log variances

    all_mean = torch.cat(all_mean)
    all_labels = torch.cat(all_labels)
    all_logvar = torch.cat(all_logvar)  # Concatenate log variances

    plt.figure(figsize=(6, 4))
    x = all_mean[:, 0].numpy()
    y = all_mean[:, 1].numpy()
    scatter = plt.scatter(x, y, c=all_labels.numpy(), cmap='tab10', alpha=0.4, s=10)
    legend1 = plt.legend(*scatter.legend_elements(), title="Cat", fontsize=8)
    plt.gca().add_artist(legend1)
    plt.xlabel('Latent Dimension 1', fontsize=8)
    plt.ylabel('Latent Dimension 2', fontsize=8)
    plt.title(f'{data_name.upper()} Data Latent Space Scatter Plot', fontsize=10)
    plt.grid(True, which='both')
    plt.savefig(os.path.join(here, f'latent_space_scatter_{data_name}.png'))
    plt.show()

    # Compute covariance matrix
    # calculates a scaled version of the covariance matrix of the means. By multiplying each element in the covariance
    # matrix by the corresponding mean variance, we're essentially adjusting for the scales of the different latent
    # dimensions. This scaling doesn't directly give us the correlation matrix, but it does make the values in the
    # matrix more comparable across dimensions.
    all_var = torch.exp(all_logvar)  # 59904x10
    cov_mean_ = torch.cov(all_mean.T)
    mean_var_ = torch.mean(all_var, dim=0)
    covariance_matrix = cov_mean_ * mean_var_
    print(covariance_matrix)
    correlation_matrix = torch.corrcoef(all_mean.T)
    print(correlation_matrix)

    import seaborn as sns

    sns.set(font_scale=0.8)
    # Plot heatmap of covariance matrix
    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, annot_kws={"size": 8})
    plt.title(f"Latent Space Covariance Matrix (Data:{data_name.upper()})", fontsize=9)
    plt.xlabel("Latent Dimension", fontsize=8)
    plt.ylabel("Latent Dimension", fontsize=8)
    plt.savefig(os.path.join(here, f'img/latent_space_covariance_{data_name}.png'))
    plt.show()

    plt.figure(figsize=(4.8, 4.2))
    sns.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, annot_kws={"size": 8})
    plt.title(f"Latent Space Correlation Coefficients Matrix (Data:{data_name.upper()})", fontsize=9)
    plt.xlabel("Latent Dimension", fontsize=8)
    plt.ylabel("Latent Dimension", fontsize=8)
    plt.savefig(os.path.join(here, f'img/latent_space_coefficient_{data_name}.png'))
    plt.show()

    return all_mean

from sklearn.manifold import TSNE

def plot_latent_space_tsne(model, dataloader, data_name, device):
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

    # Apply t-SNE to reduce the dimensionality to 2
    tsne = TSNE(n_components=2, random_state=42)
    all_mean_2d = tsne.fit_transform(all_mean)

    # Plot the t-SNE result
    plt.figure(figsize=(8, 6))
    x = all_mean_2d[:, 0]
    y = all_mean_2d[:, 1]
    scatter = plt.scatter(x, y, c=all_labels.numpy(), cmap='tab10', alpha=0.4, s=10)
    legend1 = plt.legend(*scatter.legend_elements(), title="Cat", fontsize=8)
    plt.gca().add_artist(legend1)
    plt.xlabel('t-SNE Dimension 1', fontsize=8)
    plt.ylabel('t-SNE Dimension 2', fontsize=8)
    plt.title(f'{data_name.upper()} Data Latent Space (t-SNE) Scatter Plot', fontsize=10)
    plt.grid(True, which='both')
    plt.savefig(os.path.join(here, f'latent_space_tsne_{data_name}.png'))
    plt.show()

    return all_mean

# Example call (to be placed in your main function or where appropriate):
# latent_vectors = plot_latent_space_tsne(model, valid_loader, 'mnist', device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, codings_size = load_checkpoint(args.checkpoint_path, device)
    generate_images(model, device, num_images=args.num_images, codings_size=codings_size)
    semantic_interpolation(model, device, codings_size=codings_size, num_steps=args.num_steps)

    DATA_PATH = Path(args.data_path)
    train_loader, valid_loader = load_data(DATA_PATH, args.batch_size)
    latent_vectors_train = plot_latent_space(model, train_loader, "train", device)
    plot_latent_space(model, valid_loader, "valid", device)

    # Plot histograms of latent dimensions
    plot_histograms(latent_vectors_train)

    plot_latent_space_tsne(model, valid_loader, 'valid', device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with a trained Variational Autoencoder")
    parser.add_argument('--checkpoint_path', type=str, default=f"{here}/mbin/best.pth", help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default=DS_PATH_FASHION_MNIST, help='Path to dataset')
    parser.add_argument('--num_images', type=int, default=12, help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')

    args = parser.parse_args()
    main(args)
