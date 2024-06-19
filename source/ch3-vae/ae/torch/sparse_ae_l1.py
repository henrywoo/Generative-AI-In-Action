import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from config import save_fig
import matplotlib.pyplot as plt
from data import load_data, generate_sparsity_plot, plot_reconstructions
from hiq import print_model, deterministic


class SparseL1Autoencoder(nn.Module):
    """
    🌳 SparseL1Autoencoder<all params:218084>
    ├── Sequential(encoder)
    │   ├── Linear(1)|weight[100,784]|bias[100]
    │   └── Linear(3)|weight[300,100]|bias[300]
    └── Sequential(decoder)
        ├── Linear(0)|weight[100,300]|bias[100]
        └── Linear(2)|weight[784,100]|bias[784]
    """

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
            nn.Unflatten(1, (1, 28, 28))
        )
        self.sigmoid_output = None

    def forward(self, x):
        x = self.encoder(x)
        self.sigmoid_output = x
        x = self.decoder(x)
        return x


def l1_regularization(output, l1_lambda):
    l1_norm = output.abs().sum()
    return l1_lambda * l1_norm


def train(model, criterion, optimizer, train_loader, val_loader, num_epochs, l1_lambda, checkpoint_path):
    train_losses = []
    valid_losses = []
    train_recon_losses = []
    train_l1_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        running_l1_loss = 0.0
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            l1_loss = l1_regularization(model.sigmoid_output, l1_lambda)
            recon_loss = criterion(outputs, X_batch)
            loss = recon_loss + l1_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_l1_loss += l1_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_recon_loss = running_recon_loss / len(train_loader)
        avg_train_l1_loss = running_l1_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon_loss)
        train_l1_losses.append(avg_train_l1_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                outputs = model(X_batch)
                val_loss = criterion(outputs, X_batch)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        valid_losses.append(avg_val_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}, Recon Loss: {avg_train_recon_loss:.4f}, L1 Loss: {avg_train_l1_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Save checkpoint
        if epoch == num_epochs - 1:
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_recon_losses': train_recon_losses,
                'train_l1_losses': train_l1_losses
            }, checkpoint_path)

    return train_losses, valid_losses, train_recon_losses, train_l1_losses


def plot_losses(train_losses, valid_losses, train_recon_losses, train_l1_losses):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Total Train Loss', marker='o')
    plt.plot(valid_losses, label='Validation Loss', marker='x')
    plt.plot(train_recon_losses, label='Train Recon Loss', marker='v')
    plt.plot(train_l1_losses, label='Train L1 Loss', marker='H')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses over Epochs')
    save_fig("sparse_ae_l1_loss")
    plt.show()


def main(args):
    # Load datasets
    train_loader, val_loader, test_loader = load_data(args.dataset, args.batch_size)

    # Instantiate the model
    model = SparseL1Autoencoder()
    print_model(model)

    # Load checkpoint if available
    checkpoint_path = f"mbin/{args.dataset}_sparse_ae_l1.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        train_recon_losses = checkpoint['train_recon_losses']
        train_l1_losses = checkpoint['train_l1_losses']
        print(f"Loaded checkpoint from {checkpoint_path}")

        # Plot losses
        plot_losses(train_losses, valid_losses, train_recon_losses, train_l1_losses)
    else:
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.NAdam(model.parameters())

        # Train the model
        train_losses, valid_losses, train_recon_losses, train_l1_losses = train(model, criterion, optimizer,
                                                                                train_loader, val_loader, args.epochs,
                                                                                args.l1_lambda, checkpoint_path)

        # Plot losses
        plot_losses(train_losses, valid_losses, train_recon_losses, train_l1_losses)

    # Plot reconstructions
    plot_reconstructions(model, test_loader)

    # Save sparsity plot
    generate_sparsity_plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse L1 autoencoder.")
    parser.add_argument('--dataset', type=str, default="fashion_mnist", choices=['fashion_mnist', 'celeba'],
                        help="Dataset to use ('fashion_mnist' or 'celeba').")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help="Weight for the L1 regularization.")
    args = parser.parse_args()
    main(args)
