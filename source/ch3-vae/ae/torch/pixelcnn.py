import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *  # Assuming you have a config file for hyperparameters

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")

        _, _, kh, kw = self.weight.size()
        self.register_buffer("mask", torch.zeros_like(self.weight))
        self.mask[:, :, :kh // 2, :] = 1
        self.mask[:, :, kh // 2, :kw // 2] = 1
        if mask_type == "B":
            self.mask[:, :, kh // 2, kw // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=1)
        self.masked_conv = MaskedConv2d("B", filters, filters // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters // 2, filters, kernel_size=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.masked_conv(x))
        x = self.conv2(x)
        return F.relu(x + residual)  # Ensure output is activated



def get_pixelcnn(pixelcnn_input_shape, num_embeddings):
    num_residual_blocks = 2
    num_pixelcnn_layers = 2
    in_channels = pixelcnn_input_shape[0] * num_embeddings

    layers = [
        MaskedConv2d("A", in_channels, in_channels, kernel_size=pixelcnn_input_shape[1], padding="same"),
        nn.ReLU(),
    ]
    for _ in range(num_residual_blocks):
        layers.append(ResidualBlock(in_channels))
    for _ in range(num_pixelcnn_layers):
        layers.append(MaskedConv2d("B", in_channels, in_channels, kernel_size=1, stride=1, padding="valid"))
        layers.append(nn.ReLU())
    layers.append(nn.Conv2d(in_channels, num_embeddings, kernel_size=1, stride=1, padding="valid"))

    return nn.Sequential(*layers)
