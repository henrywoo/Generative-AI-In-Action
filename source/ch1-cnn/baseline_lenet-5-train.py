import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 50
SCHEDULER_ETA_MIN = 0.00001


def initialize_model():
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = LeNet5().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_STEP_SIZE,
                                                           eta_min=SCHEDULER_ETA_MIN)
    return model, optimizer, scheduler


def load_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def train_and_validate(model, train_loader, val_loader, optimizer, scheduler):
    train_losses = []
    val_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
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
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = F.cross_entropy(output, target)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        print(f'Epoch: {epoch}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, NUM_EPOCHS + 1), train_losses, marker='o', label='Training Loss', alpha=0.5)
    plt.plot(np.arange(1, NUM_EPOCHS + 1), val_losses, marker='x', label='Validation Loss', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("model_lenet5_train_val_loss.png")
    plt.show()


def main():
    model, optimizer, scheduler = initialize_model()
    train_loader, val_loader = load_data()
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, optimizer, scheduler)
    plot_losses(train_losses, val_losses)
    torch.save(model.state_dict(), "model_lenet5.pth")


if __name__ == "__main__":
    main()
