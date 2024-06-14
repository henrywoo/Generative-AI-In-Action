import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from vit_pytorch import ViT
from tqdm import tqdm
from models import TiTok
from proxy import get_proxy_codes, load_vqgan_model
from hiq.vis import print_model

import torch.nn.functional as F


def compute_loss(outputs, proxy_codes):
    # outputs: (batch_size, num_patches, codebook_dim)
    # proxy_codes: (batch_size, num_patches)
    # Reshape proxy codes to match the output shape
    proxy_codes = proxy_codes.view(outputs.size(0), -1)

    # Compute the loss (cross-entropy loss)
    loss = F.cross_entropy(outputs.permute(0, 2, 1), proxy_codes)
    return loss


# Assuming you have a function get_proxy_codes() that returns the proxy codes
def warmup_training(model, dataloader, optimizer, num_epochs, vqgan_model):
    model.train()
    for epoch in range(num_epochs):
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs, ids = model(images)
            # Calculate the loss with proxy codes
            proxy_codes = get_proxy_codes(images, vqgan_model)
            loss = compute_loss(outputs, proxy_codes)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def finetune_training(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.codebook.parameters():
        param.requires_grad = False

    for epoch in range(num_epochs):
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate the loss
            loss = criterion(outputs, images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Average Test Loss: {avg_loss:.4f}')


if __name__ == "__main__":
    # Define the training parameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs_warmup = 10
    num_epochs_finetune = 10
    latent_dim = 16
    image_size = 256
    patch_size = 32

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TiTok(
        image_size=image_size, patch_size=patch_size, num_classes=1000, dim=latent_dim, depth=6, heads=16, mlp_dim=2048
    ).to(device)
    print_model(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    vqgan_model = load_vqgan_model('VQGAN/model.yaml', 'VQGAN/last.ckpt').to(device)

    # Perform warm-up training
    warmup_training(model, train_loader, optimizer, num_epochs_warmup, vqgan_model)

    # Perform fine-tuning
    finetune_training(model, train_loader, criterion, optimizer, num_epochs_finetune)

    # Evaluate the model
    evaluate_model(model, test_loader)
