import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define transformations
transform = transforms.Compose([
    transforms.Resize(224),  # Resizing to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing with MNIST mean and std
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def modify_resnet18():
    from hiq.vis import print_model
    model = resnet18(pretrained=True)
    # Adjust the first convolutional layer to accept 3-channel inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Change the output layer to have 10 outputs (for MNIST's 10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    print_model(model)
    return model

model = modify_resnet18().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_and_save_model(model, train_loader, criterion, optimizer, num_epochs=10, save_path='mnist_resnet18.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    losses = []  # List to store the average loss per epoch

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        losses.append(average_loss)
        print(f'Epoch {epoch+1}, Loss: {average_loss}')

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Plotting the loss vs epochs
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_vs_epochs.png')
    plt.show()


# Parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

# Call the training and saving function
train_and_save_model(model, train_loader, criterion, optimizer, num_epochs)

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Accuracy of the network on test images: {100 * correct / total} %')

evaluate_model(model, test_loader)

