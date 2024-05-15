import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
from unet import UNet
from torchvision.datasets import VOCSegmentation
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import copy


def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_set = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform,
                                target_transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_set = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform,
                              target_transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, epochs, lr, patience):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA Setup
    ema_model = copy.deepcopy(model)
    ema_decay = 0.999

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())
            loss.backward()
            optimizer.step()

            # Update EMA model
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = ema_model(images)  # Use EMA weights for validation
                loss = criterion(outputs, masks.squeeze(1).long())
                val_running_loss += loss.item()

        epoch_val_loss = val_running_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch + 1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

        scheduler.step()

        # Early stopping and saving best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(ema_model.state_dict(), 'seg_best.pth')
            print("Saved best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    # Plot training and validation loss curves
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, marker='o', label='Training Loss')
    plt.plot(val_losses, marker='x', label='Validation Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss_curve.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=21)  # PASCAL VOC has 20 classes + background
    train_loader, val_loader = get_dataloader(batch_size=args.batch_size)
    train_model(model, train_loader, val_loader, device, args.epochs, args.lr, args.patience)
