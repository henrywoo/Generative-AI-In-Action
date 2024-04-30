import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from torch.utils.data import random_split




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

# Split training dataset into training and validation sets 2-8 rule
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
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
def train_and_save_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path='mnist_resnet18_best.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training_losses = []  # List to store the average loss per epoch
    validation_losses = []  # List to store the validation loss per epoch
    best_loss = float('inf')
    patience = 30
    stale_epochs = 0

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
        training_losses.append(average_loss)

        # Validation loss calculation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss = criterion(output, target)
                total_val_loss += val_loss.item()
        average_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(average_val_loss)

        print(f'Epoch {epoch+1}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}')

        # Saving the best model
        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'New best model saved at Epoch {epoch+1} with Validation Loss: {average_val_loss}')
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f'Early stopping invoked at Epoch {epoch+1}')
                break

    # Plotting the training and validation losses
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    # Inside your train_and_save_model function, update the plotting section:
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker='o', linestyle='-', color='blue',
             label='Training Loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, marker='x', linestyle='-', color='red',
             label='Validation Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_vs_epochs.png')
    plt.show()


# Parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 60

# Call the training and saving function
train_and_save_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

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

