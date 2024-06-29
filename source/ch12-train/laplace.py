import torch
import torch.nn as nn
from torch.distributions import Laplace


class LogitLaplaceLoss(nn.Module):
    def __init__(self, scale=0.1):  # Adjust 'scale' for your specific data
        super(LogitLaplaceLoss, self).__init__()
        self.scale = scale

    def forward(self, recon_x, x):
        """
        recon_x: Reconstructed pixel values (logits) from VAE decoder
        x: Original pixel values (in range [0, 1])
        """

        # Convert original pixel values to logits
        x_logit = torch.logit(x.clamp(min=1e-7, max=1 - 1e-7))  # Avoid log(0) or log(1)

        # Create Laplace distribution for each pixel
        laplace = Laplace(loc=recon_x, scale=self.scale)

        # Calculate negative log-likelihood (NLL) under Laplace distribution
        nll = -laplace.log_prob(x_logit)

        # Average NLL over all pixels
        loss = nll.mean()

        return loss


# Example Usage:
recon_image = torch.randn(1, 3, 256, 256)  # Example reconstructed image (logits)
original_image = torch.rand(1, 3, 256, 256)  # Example original image (in range [0, 1])

criterion = LogitLaplaceLoss()
loss = criterion(recon_image, original_image)

print("Logit-Laplace Loss:", loss.item())
