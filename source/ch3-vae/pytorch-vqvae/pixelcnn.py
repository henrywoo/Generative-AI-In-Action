import torch
import torch.nn as nn
import torch.nn.functional as F

"""
In the context of using PixelCNN to model the latent space of a VectorQuantizedVAE, the typical setup involves predicting the discrete indices of the latent embeddings. Therefore, the output of PixelCNN should be logits over the possible discrete embeddings (i.e., classification task).

PixelCNN Output and Loss

PixelCNN Output: The output of PixelCNN should be logits for each pixel in the latent space over the possible discrete embeddings. Therefore, if num_embeddings = 512, the output shape will be (batch_size, num_embeddings, height, width).
Loss Calculation: The targets for the loss function should be the discrete indices of the latent embeddings. These indices are obtained during the quantization step of the VQ-VAE.
"""
class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(PixelCNN, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(128, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class AutoregressivePixelCNN(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(AutoregressivePixelCNN, self).__init__()
        self.conv1 = MaskedConv2d('A', embed_dim, 128, kernel_size=7, padding=3)
        self.conv2 = MaskedConv2d('B', 128, 128, kernel_size=7, padding=3)
        self.conv3 = MaskedConv2d('B', 128, 128, kernel_size=7, padding=3)
        self.conv4 = MaskedConv2d('B', 128, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Example usage:
# pixelcnn_model = AutoregressivePixelCNN(num_embeddings=512, embed_dim=256)
