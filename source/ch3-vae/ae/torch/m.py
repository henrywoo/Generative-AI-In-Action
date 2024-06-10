from config import *
import torch.nn.functional as F
from vqvae_plot import plot_original_vs_code, show_all_subplots  # Make sure this module is available

LATENT_DIM = 16
NUM_EMBEDDINGS = 64
BATCH_SIZE = 128


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.beta * e_latent_loss

        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


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
        quantized, vq_loss, encoding_indices = self.quantizer(encoded)
        reconstructions = self.decoder(quantized)
        return reconstructions, vq_loss, encoding_indices


# Train the VQVAE model
def train_vqvae(model, train_loader, train_variance, num_epochs=30):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_losses = []
    recon_losses = []
    vq_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch = 0
        recon_loss_epoch = 0
        vq_loss_epoch = 0

        # Wrap the train_loader with tqdm
        for data, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            data = data.to(device).float() / 255.0 - 0.5
            optimizer.zero_grad()
            reconstructions, vq_loss, encoding_indices = model(data)
            recon_loss = F.mse_loss(reconstructions, data) / train_variance
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            # print(f"reconstruction loss: {recon_loss}, vq loss: {vq_loss}")
            total_loss_epoch += loss.item()
            recon_loss_epoch += recon_loss.item()
            vq_loss_epoch += vq_loss.item()

        total_losses.append(total_loss_epoch / len(train_loader))
        recon_losses.append(recon_loss_epoch / len(train_loader))
        vq_losses.append(vq_loss_epoch / len(train_loader))

        print(f'Epoch {epoch + 1}, Average Loss: {total_losses[-1]}')

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), total_losses, label='Total Loss')
    plt.plot(range(1, num_epochs + 1), recon_losses, label='Reconstruction Loss')
    plt.plot(range(1, num_epochs + 1), vq_losses, label='VQ Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Function to load or train VQVAE
def load_or_train_vqvae():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute the variance of the training data; train_data.shape: (60000, 28, 28)
    train_data = train_dataset.data.numpy().astype(np.float32) / 255.0 - 0.5
    train_variance = np.var(train_data) # 0.09493039

    model_path = Path("mbin/vqvae")
    model_path.mkdir(parents=True, exist_ok=True)
    model_file = model_path / "vqvae_model.pth"
    vqvae_model = VQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)

    if model_file.exists():
        vqvae_model.load_state_dict(torch.load(model_file))
    else:
        train_vqvae(vqvae_model, train_loader, train_variance, num_epochs=30)
        torch.save(vqvae_model.state_dict(), model_file)

    return vqvae_model, train_loader, test_loader


# Use the function to load or train VQVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vqvae_model, train_loader, test_loader = load_or_train_vqvae()

# Randomly select test images
test_images = []
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        test_images.append(data.cpu().numpy())
test_images = np.concatenate(test_images, axis=0)
idx = np.random.choice(len(test_images), 10, replace=False)
selected_test_images = test_images[idx]
selected_test_images_tensor = torch.tensor(selected_test_images).to(device)

# Reconstruct the selected test images
with torch.no_grad():
    reconstructions, _, _ = vqvae_model(selected_test_images_tensor)
reconstructions = reconstructions.cpu().numpy()

# Show original vs reconstructed images
show_all_subplots(selected_test_images, reconstructions)

# Generate the codebook indices for the selected test images
with torch.no_grad():
    encoded_outputs = vqvae_model.encoder(selected_test_images_tensor).cpu().numpy()
flat_enc_outputs = torch.tensor(encoded_outputs).view(-1, LATENT_DIM).to(device)

# Get codebook indices
codebook_indices = vqvae_model.quantizer.get_code_indices(flat_enc_outputs).cpu().numpy()

# Reshape codebook indices correctly
batch_size, channels, height, width = encoded_outputs.shape
codebook_indices = codebook_indices.reshape(batch_size, height, width)

# Plot original vs codebook indices using selected test images
plot_original_vs_code(selected_test_images, codebook_indices)

