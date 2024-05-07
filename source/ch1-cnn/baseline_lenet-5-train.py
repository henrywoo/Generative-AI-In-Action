import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Check if CUDA is available and set device to GPU if it is, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)  # Adjust from 16*4*4 to 16*5*5
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)  # Change here to match the new dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Define the transformations for the MNIST images
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 to provide enough padding for LeNet (originally 28x28)
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training data
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# Split the MNIST dataset into training and validation sets
train_size = int(0.9 * len(train_set))
val_size = len(train_set) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

def train_and_validate(model, device, train_loader, val_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch: {epoch}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
    return train_loss, val_loss


model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters())
num_epochs = 20
train_losses = []
val_losses = []

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

for epoch in range(1, num_epochs + 1):
    train_loss, val_loss = train_and_validate(model, device, train_loader, val_loader, optimizer, epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step()

# Save the model
torch.save(model.state_dict(), "model_lenet5.pth")

# Plotting the training and validation loss
plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, num_epochs+1), train_losses, marker='o', label='Training Loss', alpha=0.5)
plt.plot(np.arange(1, num_epochs+1), val_losses, marker='x', label='Validation Loss', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
output_path = "model_lenet5_train_val_loss.png"
plt.savefig(output_path)
plt.show()


