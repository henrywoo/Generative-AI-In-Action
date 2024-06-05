import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import train_loader

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        # Flatten input
        flat_x = x.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and calculate loss
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = x + (quantized - x).detach()
        return loss, quantized

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.conv_trans1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = torch.sigmoid(self.conv_trans3(x))
        return x

class VQVAE(nn.Module):
    def __init__(self, latent_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon

def train_vqvae(model, data_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for img, _ in data_loader:
            img = img.to(device)
            optimizer.zero_grad()
            vq_loss, img_recon = model(img)
            recon_loss = F.mse_loss(img_recon, img)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss {overall_loss/len(data_loader):.4f}')
    return model



