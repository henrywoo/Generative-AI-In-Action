import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
from stanford_dogs import StanfordDogsDataset




def show_sample_image_from_stanford_dogs():
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Initialize the dataset
    dataset = StanfordDogsDataset(images_dir='/media/wukong/datafat/data/stanford_dogs/images/Images',
                                  annotations_dir='/media/wukong/datafat/data/stanford_dogs/annotations/Annotation',
                                  transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get a random sample from the dataset
    sample_image, sample_label = next(iter(dataloader))

    # Denormalize the image for displaying
    sample_image = sample_image * 0.5 + 0.5  # Unnormalize the image

    # Convert the image to numpy array
    sample_image_np = sample_image.squeeze().permute(1, 2, 0).numpy()

    # Plot the image
    plt.imshow(sample_image_np)
    plt.title(f"Label: {dataset.breeds[sample_label]}")
    plt.axis('off')
    plt.show()


# Define a simple neural network
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

data_dir = os.getenv('DEVROOT2') + "/data/stanford_dogs"
def train_model():
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Initialize the dataset and dataloader
    train_dataset = StanfordDogsDataset(images_dir=f'{data_dir}/images/Images',
                                        annotations_dir=f'{data_dir}/annotations/Annotation',
                                        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the neural network, loss function, and optimizer
    num_classes = len(train_dataset.breeds)
    model = SimpleCNN(num_classes=num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")

if __name__ == '__main__':
    # Call the function to display a sample image
    show_sample_image_from_stanford_dogs()
    # Train the model
    train_model()
