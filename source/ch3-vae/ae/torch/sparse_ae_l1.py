import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from data import load_data, generate_sparsity_plot, plot_reconstructions


class SparseL1Autoencoder(nn.Module):
    def __init__(self):
        super(SparseL1Autoencoder, self).__init__()
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
            nn.Unflatten(1, (28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm


def train(model, criterion, optimizer, train_loader, val_loader, num_epochs, l1_lambda, checkpoint_path):
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, X_batch) + l1_regularization(model, l1_lambda)
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


def main(args):
    # Load datasets
    train_loader, val_loader, test_loader = load_data(args.dataset, args.batch_size)

    # Instantiate the model
    model = SparseL1Autoencoder()

    # Load checkpoint if available
    checkpoint_path = f"{args.dataset}_sparse_l1.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.NAdam(model.parameters())

        # Train the model
        train_losses, valid_losses = train(model, criterion, optimizer, train_loader, val_loader, args.epochs,
                                           args.l1_lambda, checkpoint_path)

    # Plot reconstructions
    plot_reconstructions(model, test_loader)

    # Save sparsity plot
    generate_sparsity_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse L1 autoencoder.")
    parser.add_argument('--dataset', type=str, default="fashion_mnist", choices=['fashion_mnist', 'celeba'],
                        help="Dataset to use ('fashion_mnist' or 'celeba').")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help="Weight for the L1 regularization.")
    args = parser.parse_args()
    main(args)
