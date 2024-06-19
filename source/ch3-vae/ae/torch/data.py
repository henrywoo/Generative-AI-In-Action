from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import torch
from config import save_fig
import matplotlib.pyplot as plt

def load_data(dataset, batch_size):
    if dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    elif dataset == 'celeba':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.CelebA(root='data', split='train', download=True, transform=transform)
        test_dataset = datasets.CelebA(root='data', split='test', download=True, transform=transform)

    else:
        raise ValueError("Unsupported dataset. Choose from 'fashion_mnist' or 'celeba'.")

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def generate_sparsity_plot():
    plt.style.use('ggplot')
    p = 0.1
    q = np.linspace(0.001, 0.999, 500)
    kl_div = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    mse = (p - q) ** 2
    mae = np.abs(p - q)
    plt.plot([p, p], [0, 0.3], "k:")
    plt.text(0.05, 0.32, "Target\nsparsity", fontsize=14)
    plt.plot(q, kl_div, "b-", label="KL divergence")
    plt.plot(q, mae, "g--", label=r"MAE ($\ell_1$)")
    plt.plot(q, mse, "r--", linewidth=1, label=r"MSE ($\ell_2$)")
    plt.legend(loc="upper left", fontsize=14)
    plt.xlabel("Actual sparsity")
    plt.ylabel("Cost", rotation=0)
    plt.axis([0, 1, 0, 0.95])
    plt.grid(True)
    save_fig("sparsity_loss_plot.png")
    plt.show()


def plot_reconstructions(model, test_loader):
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(X_batch[0].reshape(28, 28), cmap='gray')
            axes[0].set_title('Original')
            axes[1].imshow(outputs[0].reshape(28, 28), cmap='gray')
            axes[1].set_title('Reconstructed')
            plt.show()
            break


def plot_reconstructions(model, test_loader, n_images=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of images from the test_loader
        images, _ = next(iter(test_loader))
        images = images[:n_images]

        # Get the reconstructions
        reconstructions = model(images)

        # Clip the reconstructions to the range [0, 1]
        reconstructions = torch.clamp(reconstructions, 0, 1)

        fig = plt.figure(figsize=(n_images * 1.5, 3))
        for image_index in range(n_images):
            plt.subplot(2, n_images, 1 + image_index)
            plt.imshow(images[image_index].cpu().numpy().reshape(28, 28))
            plt.axis("off")
            plt.subplot(2, n_images, 1 + n_images + image_index)
            plt.imshow(reconstructions[image_index].cpu().numpy().reshape(28, 28))
            plt.axis("off")
        plt.show()