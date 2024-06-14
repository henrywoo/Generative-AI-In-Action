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

import torch.optim as optim
from tqdm import tqdm


def warmup_training(model, dataloader, optimizer, num_epochs):
    model.train()
    mse_loss = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, quantized_tokens = model(images)
            loss = mse_loss(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


if __name__ == "__main__":
    # Define the training parameters
    batch_size = 64
    learning_rate = 1e-4
    num_epochs_warmup = 10
    num_epochs_finetune = 10
    latent_dim = 256
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    vqgan_model = load_vqgan_model('VQGAN/model.yaml', 'VQGAN/last.ckpt').to(device)

    model = TiTok(
        image_size=image_size,
        patch_size=patch_size,
        dim=latent_dim,
        depth=6,
        heads=16,
        mlp_dim=2048,
        K=8,
        codebook=vqgan_model.quantize.embedding.weight,
    ).to(device)

    print_model(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    warmup_training(model, train_loader, optimizer, num_epochs_warmup)
