import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch.nn.functional as F
from data import load_data, generate_sparsity_plot#, plot_reconstructions


class KLDivergenceRegularizer:
    def __init__(self, weight, target):
        self.weight = weight
        self.target = target

    def __call__(self, inputs):
        mean_activities = torch.mean(inputs, dim=0)
        kl_div = F.kl_div(mean_activities, self.target, reduction='batchmean')
        return self.weight * (kl_div + F.kl_div(1. - mean_activities, 1. - self.target, reduction='batchmean'))


class SparseKLAutoencoder(nn.Module):
    def __init__(self, kld_reg):
        super(SparseKLAutoencoder, self).__init__()
        self.kld_reg = kld_reg
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 28 * 28),
            nn.Unflatten(1, (1, 28, 28))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def regularization_loss(self, X_train):
        encoded = self.encoder(X_train)
        return self.kld_reg(encoded)


def train(model, criterion, optimizer, train_loader, val_loader, num_epochs, checkpoint_path):
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch) + model.regularization_loss(X_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                outputs = model(X_batch)
                val_loss = criterion(outputs, X_batch)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        valid_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    return train_losses, valid_losses


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

def main(args):
    # Load datasets
    train_loader, val_loader, test_loader = load_data(args.dataset, args.batch_size)

    # Instantiate the model
    kld_reg = KLDivergenceRegularizer(weight=args.kld_weight, target=torch.tensor(args.kld_target))
    model = SparseKLAutoencoder(kld_reg)

    # Load checkpoint if available
    checkpoint_path = f"{args.dataset}_sparse_ae_kl.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.NAdam(model.parameters())

        # Train the model
        train_losses, valid_losses = train(model, criterion, optimizer, train_loader, val_loader, args.epochs,
                                           checkpoint_path)

    # Plot reconstructions
    plot_reconstructions(model, test_loader)

    # Save sparsity plot
    generate_sparsity_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse KL autoencoder.")
    parser.add_argument('--dataset', type=str, default="fashion_mnist", choices=['fashion_mnist', 'celeba'],
                        help="Dataset to use ('fashion_mnist' or 'celeba').")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--kld_weight', type=float, default=5e-3, help="Weight for the KL divergence regularization.")
    parser.add_argument('--kld_target', type=float, default=0.1,
                        help="Target sparsity for KL divergence regularization.")
    args = parser.parse_args()
    main(args)
