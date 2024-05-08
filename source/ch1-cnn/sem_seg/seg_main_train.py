import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from unet import UNet



from torchvision.datasets import VOCSegmentation


def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Download and load the training set
    train_set = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform,
                                target_transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Download and load the validation set
    val_set = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform,
                              target_transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs=50):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())  # Ensure mask dimensions and types are correct
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Evaluate on validation set
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1).long())
                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'seg_best.pth')
            print("Saved best model")

    # Plot training and validation loss curves
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', label='Training Loss', alpha=0.5)
    plt.plot(val_losses, marker='x', label='Validation Loss', alpha=0.5)
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss_curve.png')
    plt.show()


if __name__ == '__main__':
    # Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=21)  # PASCAL VOC has 20 classes + background
    from hiq.vis import print_model
    print_model(model)
    train_loader, val_loader = get_dataloader(batch_size=4)
    train_model(model, train_loader, val_loader, device)

