import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a simple neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    return train_loader

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Main function to execute the training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = load_data()
    for epoch in range(1, 11):  # Train for 10 epochs
        train(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Training complete and model saved.')

if __name__ == '__main__':
    main()
