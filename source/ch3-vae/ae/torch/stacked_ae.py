from config import *

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from hiq import set_seed
import matplotlib.pyplot as plt

set_seed(has_torch=True)

# Load and preprocess the Fashion MNIST dataset
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
class StackedAutoencoderBAD(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Unflatten(1, (28, 28))
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class StackedAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.e1 = nn.Linear(784, 100)
        self.e2 = nn.Linear(100, 30)
        # Decoder
        self.d1 = nn.Linear(30, 100)
        self.d2 = nn.Linear(100, 784)
    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.d1(x))
        x = self.d2(x)
        x = x.reshape(-1, 1, 28, 28)
        return x

# Instantiate the autoencoder
autoencoder = StackedAutoencoder()

# Define the optimizer and loss function
optimizer = optim.NAdam(autoencoder.parameters())
criterion = nn.MSELoss()

# Train the autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    autoencoder.train()
    for batch in train_loader:
        inputs, _ = batch
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

    autoencoder.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            inputs, _ = batch
            outputs = autoencoder(inputs)
            valid_loss += criterion(outputs, inputs).item()
    valid_loss /= len(valid_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}")

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

# Get some validation images
valid_images, _ = next(iter(valid_loader))
plot_reconstructions(autoencoder, valid_images, n_images=5)
save_fig("reconstruction_plot")
plt.show()
