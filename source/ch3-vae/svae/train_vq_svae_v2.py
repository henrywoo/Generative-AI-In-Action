from train_svae_mnist import *

VQ_LOSS_WEIGHT = 10
intermediate_dim = 256
CONTRASTIVE = True
CNN_NETWORK = False
BOOK_SIZE = 1024

class SoftVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.use_cosine_distance = use_cosine_distance
        self.beta = beta

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)  # Normalize on initialization

    def forward(self, inputs):
        """A convex combination of unit-norm vectors does not necessarily have unit norm. It will lie within the convex 
        hull of the unit sphere points, which means inside the unit ball but not necessarily on the surface.
        """
        # Normalize embeddings to lie on unit sphere (ensure they remain normalized)
        with torch.no_grad():
            self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)

        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        similarity = []
        if self.use_cosine_distance:
            # Normalize input for cosine distance
            flat_input = F.normalize(flat_input, p=2, dim=1)
            # Compute cosine distance (1 - cosine similarity)
            similarity = torch.matmul(flat_input, self.embeddings.weight.t())
            #distances = 1 - similarity
        else:
            # Compute Euclidean distance
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                         + torch.sum(self.embeddings.weight**2, dim=1)
                         - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Apply softmax to distances to get soft assignments
        soft_assignments = F.softmax(self.beta * similarity, dim=1)

        # Quantize and unflatten
        quantized = torch.matmul(soft_assignments, self.embeddings.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='sum')
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), soft_assignments

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
        self.vq_layer = SoftVectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance, beta)
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

class VQ_SVAE_CNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, use_cosine_distance=False, beta=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # 1x28x28 -> 6x24x24
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # 6x24x24 -> 6x12x12
            nn.Conv2d(6, 16, 5),  # 6x12x12 -> 16x8x8
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # 16x8x8 -> 16x4x4
            nn.Conv2d(16, 120, 4),  # 16x4x4 -> 120x1x1
            nn.ReLU(),
            nn.Flatten(),  # Flatten the tensor
            nn.Linear(120, 84),  # 120 -> 84
            nn.ReLU(),
            nn.Linear(84, 3)  # Map to 3D point
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, embedding_dim),  # Map from 3D point to embedding_dim
            nn.ReLU(),
            nn.Linear(embedding_dim, 128 * 7 * 7),  # Map to 128 channels with 7x7 feature maps
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),  # Unflatten to match ConvTranspose2d input shape
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 128x7x7 -> 64x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64x14x14 -> 32x28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),  # 32x28x28 -> 1x28x28
            nn.Sigmoid()  # Use Sigmoid if output values need to be in the range [0, 1]
        )
        self.vq_layer = SoftVectorQuantizer(num_embeddings, latent_dim, commitment_cost, use_cosine_distance)

    def encode(self, x):
        x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        return h

    def decode(self, z):
        return self.decoder(z)

    def quant(self, h):
        h = h.view(-1, 1, 1, 3)  # Reshape for VQ layer
        vq_loss, quantized, _ = self.vq_layer(h)
        return vq_loss, quantized

    def forward(self, x):
        h = self.encode(x)
        vq_loss, quantized = self.quant(h)
        recon_x = self.decode(quantized.view(-1, 3))  # Flatten for decoder
        recon_x = recon_x.view(x.shape[0], -1)
        return recon_x, vq_loss, quantized

    def generate(self, num_samples, device, noise_scale=0.05):
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
    if CNN_NETWORK:
        model = VQ_SVAE_CNN(num_embeddings=BOOK_SIZE, embedding_dim=latent_dim, commitment_cost=COMMITMENT_COST,
                            use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
    else:
        model = VQ_SVAE(num_embeddings=BOOK_SIZE, embedding_dim=latent_dim, commitment_cost=COMMITMENT_COST,
                       use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'vq_svae_checkpoint_v2.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = PATIENCE
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history,
              vq_loss_weight=VQ_LOSS_WEIGHT, contrastive=CONTRASTIVE)
        avg_val_loss = validate(model, test_loader, device, val_loss_history, False,
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

    plot_recon_loss(recon_loss_history, start_epoch==1, version=2)
    plot_loss(train_loss_history, val_loss_history, 2)
    visualize_latent_space(model, test_loader, device, version=2)
    visualize_reconstructed_digits(model, device, latent_dim, version=2)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1, version=2)


if __name__ == "__main__":
    wandb.init(
        project="vq_svae_v2",
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
            "use_cnn": CNN_NETWORK,
            "patience": PATIENCE,
            "contrastive": CONTRASTIVE,
            "vq_loss_type": "mse_sum",
            "hard": False,
            "margin": MARGIN
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

