from train_svae_mnist import *
from train_vq_svae_v3 import generate_spherical_points
from train_vq_svae_v2 import loss_function, train, validate

def plot_spherical_points(points):
    if points.shape[1] != 3:
        raise ValueError("Can only plot 3-dimensional points")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = points[:, 0].numpy()
    ys = points[:, 1].numpy()
    zs = points[:, 2].numpy()

    ax.scatter(xs, ys, zs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("3D Spherical Points")
    plt.show()


class HardVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False):
        super(HardVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_cosine_distance = use_cosine_distance

        # Initialize embeddings uniformly on the 3-dimensional sphere
        self.embeddings = nn.Parameter(generate_spherical_points(embedding_dim, num_embeddings))
        self.embeddings.requires_grad = False

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = F.normalize(inputs, p=2, dim=-1)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        similarity = []
        if self.use_cosine_distance:
            # Normalize input for cosine distance
            #flat_input = F.normalize(flat_input, p=2, dim=1)
            # Compute cosine distance (1 - cosine similarity)
            similarity = torch.matmul(flat_input, self.embeddings.t())
            distances = 1 - similarity
        else:
            # Compute Euclidean distance
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                         + torch.sum(self.embeddings**2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embeddings.t()))

        # Hard quantization
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = F.embedding(encoding_indices, self.embeddings).view(input_shape)

        # Use STE to pass gradients
        quantized = inputs + (quantized - inputs).detach()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)

        # Compute soft assignments for entropy calculation
        soft_assignments = F.softmax(similarity, dim=1)
        avg_probs = torch.mean(soft_assignments, dim=0)
        entropy_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        avg_similarity_entropy = torch.sum(soft_assignments * torch.log(soft_assignments + 1e-10), dim=1).mean()

        t = -entropy_loss + avg_similarity_entropy
        total_loss = e_latent_loss + self.commitment_cost * t

        # Convert quantized from BHWC -> BCHW
        q = quantized.permute(0, 3, 1, 2).contiguous()
        return total_loss, q, encoding_indices


class VQ_SVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False):
        super(VQ_SVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.vq_layer = HardVectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance)
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
        embeddings = self.vq_layer.embeddings[embedding_indices]

        # Add Gaussian noise to the embeddings
        noise = noise_scale * torch.randn_like(embeddings)
        noisy_embeddings = embeddings + noise

        # Decode the noisy embeddings to generate new images
        samples = self.decode(noisy_embeddings).cpu().data.numpy()
        return samples


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQ_SVAE(num_embeddings=512, embedding_dim=latent_dim, commitment_cost=0.25,
                    use_cosine_distance=args.use_cosine_distance).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'vq_svae_checkpoint_v4.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history)
        validate(model, test_loader, device, val_loss_history)

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }, checkpoint_path)

    plot_recon_loss(recon_loss_history, 4)
    plot_loss(train_loss_history, val_loss_history, "vq_svae_v4_loss_history.png")
    visualize_latent_space(model, test_loader, device, version=4)
    visualize_reconstructed_digits(model, device, latent_dim, version=4)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--use_cosine_distance", action="store_true", help="Use cosine distance for vector quantization")
    args = parser.parse_args()
    main(args)
