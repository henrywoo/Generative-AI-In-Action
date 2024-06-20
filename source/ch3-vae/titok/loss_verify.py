import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

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

def reconstruction_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original)

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

def show_images(original, reconstructed_perceptual, reconstructed_reconstruction):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original')
    axes[1].imshow(reconstructed_perceptual.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Perceptual Loss')
    axes[2].imshow(reconstructed_reconstruction.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Reconstruction Loss')
    plt.show()

# Assume `model_perceptual` and `model_reconstruction` are your trained models

# Load an example image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dummy data for demonstration
original = torch.rand(1, 3, 224, 224)
reconstructed_perceptual = original + torch.randn(1, 3, 224, 224) * 0.1  # Add some noise
reconstructed_reconstruction = original + torch.randn(1, 3, 224, 224) * 0.1  # Add some noise

# Calculate losses
loss_perceptual = perceptual_loss(reconstructed_perceptual, original)
loss_reconstruction = reconstruction_loss(reconstructed_reconstruction, original)

# Calculate PSNR and SSIM
psnr_perceptual = calculate_psnr(reconstructed_perceptual[0], original[0])
psnr_reconstruction = calculate_psnr(reconstructed_reconstruction[0], original[0])
ssim_perceptual = calculate_ssim(reconstructed_perceptual[0], original[0])
ssim_reconstruction = calculate_ssim(reconstructed_reconstruction[0], original[0])

# Print losses and metrics
print(f"Perceptual Loss: {loss_perceptual.item()}")
print(f"Reconstruction Loss: {loss_reconstruction.item()}")
print(f"PSNR (Perceptual): {psnr_perceptual}")
print(f"PSNR (Reconstruction): {psnr_reconstruction}")
print(f"SSIM (Perceptual): {ssim_perceptual}")
print(f"SSIM (Reconstruction): {ssim_reconstruction}")

# Show images
show_images(original[0], reconstructed_perceptual[0], reconstructed_reconstruction[0])
