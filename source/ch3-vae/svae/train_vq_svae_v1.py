from train_svae_mnist import *

VQ_LOSS_WEIGHT = 1
intermediate_dim = 256
CONTRASTIVE = True

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False,
                 enable_statistics=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_cosine_distance = use_cosine_distance
        self.enable_statistics = enable_statistics

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)

        # Buffer to track usage of each embedding
        self.register_buffer('embedding_usage', torch.zeros(self.num_embeddings))

    def forward(self, inputs):
        # Normalize embeddings to lie on unit sphere (ensure they remain normalized)
        with torch.no_grad():
            self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)

        # Normalize inputs
        inputs = F.normalize(inputs, p=2, dim=-1)

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
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(self.embeddings.weight ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Get encoding indices
        encoding_indices = torch.argmin(distances, dim=1)

        # Update embedding usage if statistics are enabled
        if self.enable_statistics:
            self.update_embedding_usage(encoding_indices)

        quantized = self.embeddings(encoding_indices).view(input_shape)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='sum')
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices

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
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
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
        embeddings = self.vq_layer.embeddings(embedding_indices)

        # Add Gaussian noise to the embeddings
        noise = noise_scale * torch.randn_like(embeddings)
        noisy_embeddings = embeddings + noise

        # Decode the noisy embeddings to generate new images
        samples = self.decode(noisy_embeddings).cpu().data.numpy()
        return samples


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQ_SVAE(num_embeddings=BOOK_SIZE, embedding_dim=latent_dim, commitment_cost=COMMITMENT_COST,
                   use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'vq_svae_checkpoint_v1.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = PATIENCE
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history,
              vq_loss_weight=VQ_LOSS_WEIGHT, contrastive=CONTRASTIVE)
        avg_val_loss = validate(model, test_loader, device, val_loss_history, True,
                                vq_loss_weight=VQ_LOSS_WEIGHT, contrastive=CONTRASTIVE)
        
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

    plot_recon_loss(recon_loss_history, start_epoch==1, version=1)
    plot_loss(train_loss_history, val_loss_history, 1)
    visualize_latent_space(model, test_loader, device, version=1)
    visualize_reconstructed_digits(model, device, latent_dim, version=1)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1, version=1)


if __name__ == "__main__":
    wandb.init(
        project="vq_svae_v1",
        config={
            "learning_rate": LR,
            "book_size": BOOK_SIZE,
            "commitment_cost": COMMITMENT_COST,
            "latent_dim": latent_dim,
            "intermediate_dim": intermediate_dim,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "beta": BETA,
            "vq_loss_weight": VQ_LOSS_WEIGHT,
            "linear_layer": 3,
            "cnn_layer": 0,
            "patience": PATIENCE,
            "contrastive": CONTRASTIVE,
            "vq_loss_type": "mse_sum",
        }
    )
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--use_cosine_distance", action="store_true", help="Use cosine distance for vector quantization")
    parser.add_argument("--beta", type=float, default=BETA, help="Beta parameter for soft quantization")
    args = parser.parse_args()
    main(args)

"""
python train_vq_svae.py --use_cosine_distance
"""