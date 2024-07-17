from train_svae_mnist import *


class HardVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0, enable_statistics=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_cosine_distance = use_cosine_distance
        self.beta = beta
        self.enable_statistics = enable_statistics

        # Initialize embeddings uniformly on the 3-dimensional sphere
        self.embeddings = nn.Parameter(generate_spherical_points(embedding_dim, num_embeddings))
        self.embeddings.requires_grad = False

        self.register_buffer('embedding_usage', torch.zeros(self.num_embeddings))

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

        # Update embedding usage if statistics are enabled
        if self.enable_statistics:
            self.update_embedding_usage(encoding_indices)

        # Use STE to pass gradients
        quantized = inputs + (quantized - inputs).detach()

        # Loss
        e_latent_loss = F.mse_loss(quantized, inputs)

        # Compute soft assignments for entropy calculation
        soft_assignments = F.softmax(self.beta * similarity, dim=1)
        avg_probs = torch.mean(soft_assignments, dim=0)
        entropy_loss = torch.sum(avg_probs * torch.log(avg_probs + 1e-10))

        avg_similarity_entropy = torch.sum(soft_assignments * torch.log(soft_assignments + 1e-10), dim=1).mean()
        t = avg_similarity_entropy - entropy_loss
        total_loss = e_latent_loss + self.commitment_cost * t

        # Convert quantized from BHWC -> BCHW
        q = quantized.permute(0, 3, 1, 2).contiguous()
        return total_loss, q, encoding_indices

    def update_embedding_usage(self, encoding_indices):
        unique_indices, counts = torch.unique(encoding_indices, return_counts=True)
        self.embedding_usage.index_add_(0, unique_indices, counts.float())

    def get_codebook_statistics(self):
        used_entries = torch.sum(self.embedding_usage > 0).item()
        usage_rate = used_entries / self.num_embeddings

        # Calculate the probabilities of each codebook entry
        probabilities = self.embedding_usage / self.embedding_usage.sum()
        # Calculate the entropy, avoiding log(0) by using only non-zero probabilities
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item()

        return {
            'usage_rate': usage_rate,
            'used_entries': used_entries,
            'entropy': entropy
        }

    def reset_statistics(self):
        self.embedding_usage.zero_()

class VQ_SVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.vq_layer = HardVectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance, beta)
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
        return recon_x, vq_loss, quantized

    def generate(self, num_samples, device, noise_scale=0.05):
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
    model = VQ_SVAE(num_embeddings=BOOK_SIZE, embedding_dim=latent_dim, commitment_cost=0.25,
                    use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
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

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history, vq_loss_weight=50)
        avg_val_loss = validate(model, test_loader, device, val_loss_history, True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break



    plot_recon_loss(recon_loss_history, start_epoch==1, version=4)
    plot_loss(train_loss_history, val_loss_history, 4)
    visualize_latent_space(model, test_loader, device, version=4)
    visualize_reconstructed_digits(model, device, latent_dim, version=4)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1, version=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--use_cosine_distance", action="store_true", help="Use cosine distance for vector quantization")
    parser.add_argument("--beta", type=float, default=BETA, help="Beta parameter for soft quantization")
    args = parser.parse_args()
    main(args)
