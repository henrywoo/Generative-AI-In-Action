import matplotlib.pyplot as plt

from train_svae_mnist import *


class SoftVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0):
        super(SoftVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_cosine_distance = use_cosine_distance
        self.beta = beta

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)  # Normalize on initialization

    def forward(self, inputs):
        # Normalize embeddings to lie on unit sphere (ensure they remain normalized)
        with torch.no_grad():
            self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)

        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        if self.use_cosine_distance:
            # Normalize input for cosine distance
            flat_input = F.normalize(flat_input, p=2, dim=1)
            # Compute cosine distance (1 - cosine similarity)
            distances = 1 - torch.matmul(flat_input, self.embeddings.weight.t())
        else:
            # Compute Euclidean distance
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                         + torch.sum(self.embeddings.weight**2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Apply softmax to distances to get soft assignments
        soft_assignments = F.softmax(-self.beta * distances, dim=1)

        # Quantize and unflatten
        quantized = torch.matmul(soft_assignments, self.embeddings.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), soft_assignments

class VQ_SVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0):
        super(VQ_SVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.vq_layer = SoftVectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return h

    def decode(self, z):
        z = z.view(-1, latent_dim)  # Flatten for decoder
        return self.decoder(z)

    def quant(self, h):
        h = h.view(-1, 1, 1, latent_dim)  # Reshape for VQ layer
        vq_loss, quantized, _ = self.vq_layer(h)
        return vq_loss, quantized

    def forward(self, x):
        h = self.encode(x.view(-1, original_dim))
        vq_loss, quantized = self.quant(h)
        recon_x = self.decode(quantized)
        return recon_x, vq_loss

    def generate(self, num_samples, device, noise_scale=0.1):
        # Sample random indices from the codebook
        embedding_indices = torch.randint(0, self.vq_layer.num_embeddings, (num_samples,), device=device)
        embeddings = self.vq_layer.embeddings(embedding_indices)

        # Add Gaussian noise to the embeddings
        noise = noise_scale * torch.randn_like(embeddings)
        noisy_embeddings = embeddings + noise

        # Decode the noisy embeddings to generate new images
        samples = self.decode(noisy_embeddings).cpu().data.numpy()
        return samples


def loss_function(recon_x, x, vq_loss):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, original_dim), reduction='sum')
    return BCE + vq_loss

def train(model, epoch, train_loader, optimizer, device, train_loss_history):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)):
        data = data.view(-1, original_dim).to(device)
        optimizer.zero_grad()
        recon_batch, vq_loss = model(data)
        loss = loss_function(recon_batch, data, vq_loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(avg_train_loss)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')

def validate(model, test_loader, device, val_loss_history):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, original_dim).to(device)
            recon_batch, vq_loss = model(data)
            test_loss += loss_function(recon_batch, data, vq_loss).item()
    avg_val_loss = test_loss / len(test_loader.dataset)
    val_loss_history.append(avg_val_loss)
    print(f'====> Test set loss: {avg_val_loss:.4f}')

def visualize_generated_images(model, num_samples, device, noise_scale=0.1):
    model.eval()
    with torch.no_grad():
        samples = model.generate(num_samples, device, noise_scale)
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
        for i in range(num_samples):
            axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
        plt.savefig("generated_images_vq_svae_v2.png")
        plt.show()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQ_SVAE(num_embeddings=512, embedding_dim=latent_dim, commitment_cost=0.25,
                    use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'vq_svae_checkpoint_v2.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history)
        validate(model, test_loader, device, val_loss_history)

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }, checkpoint_path)

    plot_loss(train_loss_history, val_loss_history, "vq_svae_v2_loss_history.png")
    visualize_latent_space(model, test_loader, device)
    visualize_reconstructed_digits(model, device, latent_dim)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument("--batch_size", type=int, default=30000, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--use_cosine_distance", action="store_true", help="Use cosine distance for vector quantization")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for soft quantization")
    args = parser.parse_args()
    main(args)
