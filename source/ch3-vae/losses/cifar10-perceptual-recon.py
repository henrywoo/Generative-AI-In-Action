# Step 1: Load the CIFAR-10 Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

# Transformations for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


# Step 2: Define a Simple Autoencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Perceptual loss function using VGG16
vgg, mean_, std_ = None, None, None

def perceptual_loss(reconstructed, original):
    global vgg, mean_, std_
    if vgg is None:
        device = reconstructed.device
        vgg = models.vgg16(weights='IMAGENET1K_V1').features
        vgg.to(device)
        vgg.eval()
        mean_ = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    reconstructed = (reconstructed - mean_) / std_
    original = (original - mean_) / std_
    features_reconstructed = vgg(reconstructed)
    features_original = vgg(original)
    loss = F.mse_loss(features_reconstructed, features_original)
    return loss

# Step 3: Training the Models
# Train the autoencoder with perceptual loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder_perceptual = SimpleAutoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder_perceptual.parameters(), lr=1e-3)

def train_perceptual(model, dataloader, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = perceptual_loss(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

train_perceptual(autoencoder_perceptual, train_loader, optimizer)

# Train the autoencoder with reconstruction loss
autoencoder_reconstruction = SimpleAutoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder_reconstruction.parameters(), lr=1e-3)

def train_reconstruction(model, dataloader, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = F.mse_loss(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

train_reconstruction(autoencoder_reconstruction, train_loader, optimizer)

# Step 4: Evaluate the Models
# Define PSNR and SSIM calculation functions
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    return ssim(img1, img2, multichannel=True)

# Test the models and compare results
def run_models(model_perceptual, model_reconstruction, dataloader):
    model_perceptual.eval()
    model_reconstruction.eval()
    psnr_perceptual, ssim_perceptual = 0, 0
    psnr_reconstruction, ssim_reconstruction = 0, 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs_perceptual = model_perceptual(inputs)
            outputs_reconstruction = model_reconstruction(inputs)
            for i in range(inputs.size(0)):
                psnr_perceptual += calculate_psnr(outputs_perceptual[i], inputs[i])
                ssim_perceptual += calculate_ssim(outputs_perceptual[i], inputs[i])
                psnr_reconstruction += calculate_psnr(outputs_reconstruction[i], inputs[i])
                ssim_reconstruction += calculate_ssim(outputs_reconstruction[i], inputs[i])
    num_images = len(dataloader.dataset)
    psnr_perceptual /= num_images
    ssim_perceptual /= num_images
    psnr_reconstruction /= num_images
    ssim_reconstruction /= num_images
    return psnr_perceptual, ssim_perceptual, psnr_reconstruction, ssim_reconstruction

psnr_perceptual, ssim_perceptual, psnr_reconstruction, ssim_reconstruction = run_models(
    autoencoder_perceptual, autoencoder_reconstruction, test_loader
)

print(f"PSNR (Perceptual): {psnr_perceptual}, SSIM (Perceptual): {ssim_perceptual}")
print(f"PSNR (Reconstruction): {psnr_reconstruction}, SSIM (Reconstruction): {ssim_reconstruction}")

# Step 5: Visual Inspection
import matplotlib.pyplot as plt

def show_images(original, reconstructed_perceptual, reconstructed_reconstruction):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed_perceptual.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Perceptual Loss')
    axes[2].imshow(reconstructed_reconstruction.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Reconstruction Loss')
    plt.show()

# Visualize results for a batch of test images
data_iter = iter(test_loader)
images, _ = data_iter.next()
images = images.to(device)

outputs_perceptual = autoencoder_perceptual(images)
outputs_reconstruction = autoencoder_reconstruction(images)

for i in range(5):  # Show 5 examples
    show_images(images[i], outputs_perceptual[i], outputs_reconstruction[i])
