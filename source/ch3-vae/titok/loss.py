import torch
import torch.nn.functional as F
from torchvision import models

vgg, mean_, std_ = None, None, None


def perceptual_loss(reconstructed, original):
    global vgg, mean_, std_
    if vgg is None:
        device = reconstructed.device
        vgg = models.vgg16(weights='IMAGENET1K_V1').features
        vgg.to(device)
        vgg.eval()
        # Normalize the images in the same way as VGG was trained
        mean_ = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_ = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    reconstructed = (reconstructed - mean_) / std_
    original = (original - mean_) / std_
    # Extract features from VGG
    features_reconstructed = vgg(reconstructed)
    features_original = vgg(original)
    # Compute the perceptual loss
    loss = F.mse_loss(features_reconstructed, features_original)
    return loss
