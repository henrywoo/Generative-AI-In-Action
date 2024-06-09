import torch
import torch.nn as nn
import torch.nn.functional as F

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