from config import *
from torch.utils.data import DataLoader, random_split
from hiq.vis import print_model

LATENT_DIM = 16
NUM_EMBEDDINGS = 128
BATCH_SIZE = 128

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_inputs = x.view(-1, self.embedding_dim)
        distances = (
                torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
                + torch.sum(self.embeddings.weight ** 2, dim=1)
                - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(x)
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = commitment_loss * self.beta + codebook_loss
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

# Define the VQVAE model
class VQVAE(nn.Module):
    def __init__(self, latent_dim=16, num_embeddings=64):
        super(VQVAE, self).__init__()
        self.encoder = self.build_encoder(latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = self.build_decoder(latent_dim)

    def build_encoder(self, latent_dim):
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1)
        )

    def build_decoder(self, latent_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss, _ = self.quantizer(encoded)
        reconstructions = self.decoder(quantized)
        return reconstructions, vq_loss

# Training function
def train_vqvae(model, dataloader, optimizer, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            reconstructions, vq_loss = model(x)
            recon_loss = F.mse_loss(reconstructions, x)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Load data and prepare DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set up device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqvae = VQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=3e-4)
print_model(vqvae)

# Train the model
train_vqvae(vqvae, train_loader, optimizer, epochs=20)
