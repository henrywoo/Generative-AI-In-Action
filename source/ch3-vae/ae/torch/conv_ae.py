import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import load_data


# Define the encoder
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # [batch, 1, 28, 28] -> [batch, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 16, 28, 28] -> [batch, 16, 14, 14]
            nn.Conv2d(16, 32, 3, padding=1),  # [batch, 16, 14, 14] -> [batch, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 32, 14, 14] -> [batch, 32, 7, 7]
            nn.Conv2d(32, 64, 3, padding=1),  # [batch, 32, 7, 7] -> [batch, 64, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [batch, 64, 7, 7] -> [batch, 64, 3, 3]
            nn.Conv2d(64, 30, 3, padding=1),  # [batch, 64, 3, 3] -> [batch, 30, 3, 3]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [batch, 30, 3, 3] -> [batch, 30, 1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # flatten


# Define the decoder
class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(30, 3 * 3 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (16, 3, 3)),
            nn.ConvTranspose2d(16, 32, 3, stride=2),  # [batch, 16, 3, 3] -> [batch, 32, 7, 7]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # [batch, 32, 7, 7] -> [batch, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            # [batch, 16, 14, 14] -> [batch, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


# Combine the encoder and decoder into an autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(torch.float32)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_loss_history, val_loss_history


# Function to plot reconstructions
def plot_reconstructions(model, data_loader, num_images=10):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(data_loader))
        outputs = model(images.to(torch.float32))
        images = images.numpy()
        outputs = outputs.numpy()

        fig, axes = plt.subplots(2, num_images, figsize=(num_images, 2))
        for i in range(num_images):
            axes[0, i].imshow(images[i].reshape(28, 28))
            axes[0, i].axis('off')
            axes[1, i].imshow(outputs[i].reshape(28, 28))
            axes[1, i].axis('off')
        plt.show()


# Function to plot loss curves
def plot_loss_curves(train_loss_history, val_loss_history):
    epochs = range(1, len(train_loss_history) + 1)
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.plot(epochs, val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def main(args):
    # Load data
    train_loader, val_loader, test_loader = load_data(args.dataset, args.batch_size)

    # Create the model
    conv_ae = ConvAutoencoder()

    # Load checkpoint if exists
    if os.path.isfile(args.model_path):
        print(f"Loading checkpoint from '{args.model_path}'")
        conv_ae.load_state_dict(torch.load(args.model_path))
    else:
        print(f"No checkpoint found at '{args.model_path}', starting from scratch.")
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.NAdam(conv_ae.parameters())

        # Train the model
        train_loss_history, val_loss_history = train_model(conv_ae, criterion, optimizer, train_loader, val_loader,
                                                           args.num_epochs)
        # Save the model
        torch.save(conv_ae.state_dict(), args.model_path)

        # Plot loss curves
        plot_loss_curves(train_loss_history, val_loss_history)

    from hiq import print_model
    print_model(conv_ae)

    # Plotting reconstructions
    plot_reconstructions(conv_ae, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Convolutional Autoencoder on Fashion MNIST or CelebA")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "celeba"],
                        help="Dataset to use for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--model_path", type=str, help="Path to save the trained model")
    args = parser.parse_args()

    if not args.model_path:
        args.model_path = f"conv_ae_{args.dataset}.pth"

    main(args)
