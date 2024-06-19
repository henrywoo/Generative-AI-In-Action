from config import *

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from hiq import set_seed
import matplotlib.pyplot as plt

set_seed(has_torch=True)

# Load and preprocess the Fashion MNIST dataset
dataset_name = 'FashionMNIST'
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split the training dataset into training and validation sets
train_size = len(train_dataset) - 5000
valid_size = 5000
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the stacked autoencoder
class StackedAutoencoder(nn.Module):
    def __init__(self, C=1):
        super().__init__()
        self.C = C
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 30),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.ReLU(),
            nn.Linear(100, 28 * 28),
            nn.Unflatten(1, (self.C, 28, 28))
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Instantiate the autoencoder
autoencoder = StackedAutoencoder()

# Define the optimizer and loss function
optimizer = optim.NAdam(autoencoder.parameters())
criterion = nn.MSELoss()

# Define checkpoint path
checkpoint_dir = './mbin'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f'ae_{dataset_name}.pt')


# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, train_losses, valid_losses, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }
    torch.save(state, path)


# Function to load checkpoint
def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        train_losses = state['train_losses']
        valid_losses = state['valid_losses']
        return epoch, train_losses, valid_losses
    else:
        return 0, [], []


# Try to load the checkpoint
start_epoch, train_losses, valid_losses = load_checkpoint(autoencoder, optimizer, checkpoint_path)

# Train the autoencoder
num_epochs = 20
for epoch in range(start_epoch, num_epochs):
    autoencoder.train()
    train_loss = 0
    for inputs, _ in train_loader:
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    autoencoder.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, _ in valid_loader:
            outputs = autoencoder(inputs)
            valid_loss += criterion(outputs, inputs).item()
    valid_loss /= len(valid_loader)
    valid_losses.append(valid_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

    # Save checkpoint after each epoch
    save_checkpoint(autoencoder, optimizer, epoch + 1, train_losses, valid_losses, checkpoint_path)


# Function to plot reconstructions
def plot_reconstructions(model, images, n_images=5):
    model.eval()
    with torch.no_grad():
        reconstructions = model(images[:n_images])
    reconstructions = reconstructions.numpy()

    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plt.imshow(images[image_index].numpy().squeeze(), cmap="binary")
        plt.axis("off")
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plt.imshow(reconstructions[image_index].squeeze(), cmap="binary")
        plt.axis("off")


# Function to plot loss over epochs
def plot_loss(train_losses, valid_losses):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(valid_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()


# Get some validation images
valid_images, _ = next(iter(valid_loader))
plot_reconstructions(autoencoder, valid_images, n_images=5)
save_fig("deep_ae_reconstruction_plot")
plt.show()

# Plot the loss over epochs
plot_loss(train_losses, valid_losses)
save_fig("deep_ae_loss_plot")
plt.show()
