import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embeddings = nn.Parameter(torch.randn(embedding_dim, num_embeddings))

    def forward(self, x):
        # Flatten input
        input_shape = x.shape
        flat_x = x.view(-1, self.embedding_dim)

        # Quantization
        encoding_indices = self.get_code_indices(flat_x)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        quantized = torch.matmul(encodings, self.embeddings.t())

        # Reshape back to input shape
        quantized = quantized.view(input_shape)

        # Calculate loss
        commitment_loss = torch.mean((quantized.detach() - x) ** 2)
        codebook_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = self.beta * commitment_loss + codebook_loss

        quantized = x + (quantized - x).detach()
        return quantized, loss

    def get_code_indices(self, flat_x):
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings ** 2, dim=0)
                     - 2 * torch.matmul(flat_x, self.embeddings))

        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, latent_dim, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, latent_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        quantized, loss = self.quantizer(z_e)
        x_recon = self.decoder(quantized)
        return x_recon, loss


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQVAE(latent_dim=16, num_embeddings=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(30):
    total_loss = 0
    for batch in train_loader:
        x, _ = batch
        x = x.to(device)
        optimizer.zero_grad()
        x_recon, loss = model(x)
        recon_loss = nn.MSELoss()(x_recon, x)
        total_loss = recon_loss + loss
        total_loss.backward()
        optimizer.step()
        total_loss += total_loss.item()
    print(f"Epoch [{epoch+1}/30], Loss: {total_loss/len(train_loader)}")

# Visualize the reconstructions
def show_subplot(original, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze().cpu().numpy() + 0.5, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze().cpu().detach().numpy() + 0.5, cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

    plt.show()

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

model.eval()
with torch.no_grad():
    for batch in test_loader:
        x, _ = batch
        x = x.to(device)
        x_recon, _ = model(x)
        for original, reconstructed in zip(x, x_recon):
            show_subplot(original, reconstructed)

        # Uncomment the break if you only want to visualize one batch
        break

# Generate the codebook indices.
model.eval()
with torch.no_grad():
    encoded_outputs = model.encoder(torch.tensor(np.expand_dims(train_dataset.data.numpy(), 1), device=device, dtype=torch.float32) / 255.0 - 0.5)
    flat_enc_outputs = encoded_outputs.view(-1, encoded_outputs.shape[-1])
    codebook_indices = model.quantizer.get_code_indices(flat_enc_outputs)
    codebook_indices = codebook_indices.view(encoded_outputs.shape[:-1]).cpu().numpy()

# Train PixelCNN
class PixelConvLayer(nn.Module):
    def __init__(self, mask_type, filters, kernel_size, activation, **kwargs):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.conv = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size // 2, **kwargs)
        self.activation = activation

    def forward(self, x):
        self.conv.weight.data *= self.mask
        x = self.conv(x)
        if self.activation == 'relu':
            x = torch.relu(x)
        return x

    def build_mask(self, kernel_size, in_channels, out_channels):
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        mask[:, :, kernel_size // 2, kernel_size // 2 + (self.mask_type == 'B'):] = 0
        mask[:, :, kernel_size // 2 + 1:] = 0
        return mask

    def build(self, input_shape):
        self.mask = self.build_mask(self.conv.kernel_size[0], input_shape[1], self.conv.out_channels).to(self.conv.weight.device)


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters // 2, 1)
        self.pixel_conv = PixelConvLayer('B', filters // 2, 3, 'relu')
        self.conv2 = nn.Conv2d(filters // 2, filters, 1)

    def forward(self, x):
        residual = x
        x = torch.relu(self.conv1(x))
        x = self.pixel_conv(x)
        x = torch.relu(self.conv2(x))
        return x + residual


class PixelCNN(nn.Module):
    def __init__(self, input_shape, num_residual_blocks, num_pixelcnn_layers, num_embeddings):
        super(PixelCNN, self).__init__()
        self.input_shape = input_shape
        self.num_embeddings = num_embeddings
        self.input_layer = PixelConvLayer('A', input_shape[0], 7, 'relu')
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_residual_blocks)])
        self.pixelcnn_layers = nn.ModuleList([PixelConvLayer('B', 128, 1, 'relu') for _ in range(num_pixelcnn_layers)])
        self.output_layer = nn.Conv2d(128, num_embeddings, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        for layer in self.pixelcnn_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


input_shape = codebook_indices.shape[1:]
pixelcnn = PixelCNN(input_shape, num_residual_blocks=2, num_pixelcnn_layers=2, num_embeddings=128).to(device)
optimizer = optim.Adam(pixelcnn.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

codebook_indices_tensor = torch.tensor(codebook_indices, dtype=torch.long, device=device)
train_loader_pixelcnn = DataLoader(codebook_indices_tensor, batch_size=128, shuffle=True)

for epoch in range(30):
    total_loss = 0
    for batch in train_loader_pixelcnn:
        optimizer.zero_grad()
        out = pixelcnn(batch)
        loss = criterion(out.view(-1, 128), batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"PixelCNN Epoch [{epoch+1}/30], Loss: {total_loss/len(train_loader_pixelcnn)}")

# Sampling from PixelCNN
def sample_from_pixelcnn(pixelcnn, shape, num_embeddings):
    with torch.no_grad():
        priors = torch.zeros(shape, device=device, dtype=torch.long)
        for row in range(shape[1]):
            for col in range(shape[2]):
                out = pixelcnn(priors)
                probs = torch.softmax(out[:, :, row, col], dim=-1)
                priors[:, row, col] = torch.multinomial(probs, 1).squeeze(-1)
        return priors

priors = sample_from_pixelcnn(pixelcnn, (10, *input_shape), 128)
priors_ohe = nn.functional.one_hot(priors, num_classes=128).float().to(device)
quantized = torch.matmul(priors_ohe, model.quantizer.embeddings.t())
quantized = quantized.view(-1, *(encoded_outputs.shape[1:]))

generated_samples = model.decoder(quantized).cpu().detach().numpy()

for i in range(generated_samples.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(priors[i].cpu().numpy())
    plt.title("Code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated_samples[i].squeeze() + 0.5)
    plt.title("Generated Sample")
    plt.axis("off")
    plt.show()
