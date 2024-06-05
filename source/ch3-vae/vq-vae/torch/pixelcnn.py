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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = MaskedConv2d('B', in_channels, in_channels//2, kernel_size=1, bias=False)
        self.conv2 = MaskedConv2d('B', in_channels//2, in_channels//2, kernel_size=3, padding=1, bias=False)
        self.conv3 = MaskedConv2d('B', in_channels//2, in_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels//2)
        self.batch_norm2 = nn.BatchNorm2d(in_channels//2)
        self.batch_norm3 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.batch_norm1(self.conv1(x)))
        out = F.relu(self.batch_norm2(self.conv2(out)))
        out = F.relu(self.batch_norm3(self.conv3(out)))
        return out + identity

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings, num_channels=128, num_layers=12):
        super(PixelCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, num_embeddings)
        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(MaskedConv2d('A', num_embeddings, num_channels, kernel_size=7, padding=3))

        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(ResidualBlock(num_channels))

        # Final layer
        self.layers.append(nn.Conv2d(num_channels, num_embeddings, kernel_size=1))

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)  # Rearrange to NCHW
        for layer in self.layers:
            x = layer(x)
        return x

def train_pixelcnn(pixelcnn, train_loader, vqvae, optimizer, device, epochs=10):
    vqvae.eval()  # Turn on evaluation mode for VQ-VAE to only use the encoder
    for epoch in range(epochs):
        for img, _ in train_loader:
            img = img.to(device)
            with torch.no_grad():
                z = vqvae.encoder(img)  # Only take the single return value
            z = z.argmax(dim=1)  # Extract indices for PixelCNN training

            optimizer.zero_grad()
            out = pixelcnn(z)
            loss = F.cross_entropy(out, z)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss {loss.item()}')
    return pixelcnn

# Assuming you have other code here to initialize your models and loaders...
