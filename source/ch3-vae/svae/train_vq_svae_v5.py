from train_svae_mnist import *
from train_vq_svae_v4 import HardVectorQuantizer
from train_vq_svae_v3 import SoftVectorQuantizer

class VQ_SVAE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes, commitment_cost, use_cosine_distance=False, beta=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = embedding_dim
        self.original_dim = original_dim
        self.encoder = nn.Sequential(
            nn.Linear(original_dim + num_classes, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, self.latent_dim)
        )
        #self.vq_layer = HardVectorQuantizer(num_embeddings, self.latent_dim, commitment_cost, use_cosine_distance, beta)
        self.vq_layer = SoftVectorQuantizer(num_embeddings, self.latent_dim, commitment_cost, use_cosine_distance, beta)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + num_classes, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )

    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        return h

    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        return self.decoder(z_c)

    def quant(self, h):
        h = h.view(-1, 1, 1, self.latent_dim)  # Reshape for VQ layer
        vq_loss, quantized, _ = self.vq_layer(h)
        return vq_loss, quantized

    def forward(self, x, c):
        h = self.encode(x.view(-1, self.original_dim), c)
        vq_loss, quantized = self.quant(h)
        recon_x = self.decode(quantized.view(-1, self.latent_dim), c)
        return recon_x, vq_loss, quantized

    def generate(self, num_samples, device, labels, noise_scale=0.05):
        # Sample random indices from the codebook
        embedding_indices = torch.randint(0, self.vq_layer.num_embeddings, (num_samples,), device=device)
        embeddings = self.vq_layer.embeddings[embedding_indices]

        # Add Gaussian noise to the embeddings
        noise = noise_scale * torch.randn_like(embeddings)
        noisy_embeddings = embeddings + noise

        # One-hot encode labels
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(device)

        # Decode the noisy embeddings to generate new images
        samples = self.decode(noisy_embeddings, one_hot_labels).cpu().data.numpy()
        return samples


def contrastive_loss_cosine(z, labels, margin=0.05):
    """Computes the contrastive loss using cosine similarity."""
    # Normalize the latent vectors
    z = F.normalize(z, p=2, dim=1)
    # Compute pairwise cosine similarity
    sim_matrix = torch.matmul(z, z.t())
    # Create label matrix
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    # Positive and negative loss
    positive_loss = label_matrix * (1 - sim_matrix)
    negative_loss = (1 - label_matrix) * F.relu(sim_matrix - margin)
    # Calculate the total contrastive loss
    contrastive_loss_value = 5 * (positive_loss + negative_loss).mean()
    return contrastive_loss_value

def contrastive_loss_eu(z, labels, margin=0.05):
    batch_size = z.size(0)
    # Compute pairwise distance matrix
    dist_matrix = torch.cdist(z, z, p=2)  # Euclidean distance
    # Create label matrix
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    # Compute contrastive loss
    positive_loss = label_matrix * dist_matrix.pow(2)
    negative_loss = (1 - label_matrix) * F.relu(margin - dist_matrix).pow(2)
    contrastive_loss_value = 5 * (positive_loss + negative_loss).sum() / (batch_size * (batch_size - 1) / 2)
    return contrastive_loss_value

def train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history, margin=0.1):
    model.train()
    train_loss = 0
    total_recon_loss = 0
    total_contrastive_loss = 0  # Initialize total contrastive loss
    contrastive_loss_value = torch.tensor(0.0, device=device)
    bss = (BATCH_SIZE * (BATCH_SIZE - 1) / 2)
    with tqdm(total=len(train_loader.dataset), desc=f"Train Epoch {epoch}", unit='samples') as pbar:
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.view(-1, original_dim).to(device)
            labels = labels.to(device)
            one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float().to(device)
            optimizer.zero_grad()
            recon_batch, vq_loss, mu = model(data, one_hot_labels)

            # Calculate contrastive loss
            z = mu.view(mu.size(0), -1)
            contrastive_loss_value = contrastive_loss_cosine(z, labels, margin)
            vq_loss *= 5
            loss, recon_loss = loss_function(recon_batch, data, vq_loss)
            loss += contrastive_loss_value
            loss.backward()
            train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss_value.item()  # Accumulate contrastive loss
            optimizer.step()
            pbar.update(data.size(0))
            pbar.set_postfix({'Loss': loss.item() / len(data), 'Recon Loss': recon_loss.item() / len(data), 'Contrastive Loss': contrastive_loss_value.item() / len(data)})

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader.dataset)  # Compute average contrastive loss
    train_loss_history.append(avg_train_loss)
    recon_loss_history.append(avg_recon_loss)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}, recon loss: {avg_recon_loss:.4f}, contrastive loss: {avg_contrastive_loss:.4f}')

def validate(model, test_loader, device, val_loss_history):
    model.eval()
    test_loss = 0
    margin = 0.1
    contrastive_loss_value = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.view(-1, original_dim).to(device)
            labels = labels.to(device)
            one_hot_labels = F.one_hot(labels, num_classes=NUM_CLASSES).float().to(device)
            recon_batch, vq_loss, mu = model(data, one_hot_labels)

            # Calculate contrastive loss
            # Reshape mu from (Batch, 1, 1, latent_dim) to (Batch, latent_dim)
            z = mu.view(mu.size(0), -1)
            contrastive_loss_value = contrastive_loss_cosine(z, labels, margin)
            vq_loss *= 50

            t, _ = loss_function(recon_batch, data, vq_loss)
            t += contrastive_loss_value
            test_loss += t.item()
    avg_val_loss = test_loss / len(test_loader.dataset)
    val_loss_history.append(avg_val_loss)
    print(f'====> Test set loss: {avg_val_loss:.4f}')
    return avg_val_loss

def visualize_latent_space(model, test_loader, device, version=0):
    model.eval()
    with torch.no_grad():
        z_means = []
        labels = []
        for data, label in test_loader:
            data = data.view(-1, original_dim).to(device)
            label = label.to(device)
            one_hot_labels = F.one_hot(label, num_classes=NUM_CLASSES).float().to(device)
            z_mean = model.encode(data, one_hot_labels)
            if hasattr(model, 'quant'):
                _, z_mean = model.quant(z_mean)
                z_mean = z_mean.view(z_mean.shape[0], latent_dim)
            z_means.append(z_mean)
            labels.append(label)
        z_means = torch.cat(z_means).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

    fig = plt.figure(figsize=(15, 15))  # Larger figure size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_means[:, 0], z_means[:, 1], z_means[:, 2], c=labels, cmap='tab10', s=10)  # Smaller points

    # Create legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)

    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')
    plt.title('Latent Space Visualization')
    plt.savefig(f"latent_space_v{version}.png")
    plt.show()


def visualize_reconstructed_digits(model, device, latent_dim, num_classes=NUM_CLASSES, version=0):
    with torch.no_grad():
        n = 10
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # Create a grid of class labels for conditioning
        class_labels = np.linspace(0, num_classes - 1, n).astype(int)
        for i in range(n):
            for j in range(n):
                z_sample = torch.randn(1, latent_dim, device=device)
                z_sample /= z_sample.norm()
                # One-hot encode the class label
                class_label = class_labels[j % num_classes]
                one_hot_label = torch.zeros(1, num_classes, device=device)
                one_hot_label[0, class_label] = 1

                # Decode with the class condition
                x_decoded = model.decode(z_sample, one_hot_label).view(digit_size, digit_size).cpu()
                digit = x_decoded.numpy()
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Reconstructed Digits')
    plt.savefig(f'reconstructed_digits_v{version}.png')
    plt.show()


def visualize_generated_images(model, num_samples_per_class, device, num_classes=NUM_CLASSES, noise_scale=0.1, version=0):
    model.eval()
    digit_size = 28
    n = num_classes
    figure = np.zeros((digit_size * n, digit_size * num_samples_per_class))

    with torch.no_grad():
        for i in range(num_classes):
            for j in range(num_samples_per_class):
                z_sample = torch.randn(1, model.latent_dim, device=device)
                z_sample /= z_sample.norm()
                # One-hot encode the class label
                one_hot_label = torch.zeros(1, num_classes, device=device)
                one_hot_label[0, i] = 1
                # Decode with the class condition
                x_decoded = model.decode(z_sample, one_hot_label).view(digit_size, digit_size).cpu()
                digit = x_decoded.numpy()
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Generated Images with Class Conditioning')
    plt.savefig(f"generated_images_vq_svae_v{version}.png")
    plt.show()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQ_SVAE(num_embeddings=512, embedding_dim=latent_dim, num_classes=NUM_CLASSES, commitment_cost=0.25,
                    use_cosine_distance=args.use_cosine_distance, beta=args.beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'vq_svae_checkpoint_v5.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history, margin=.08)
        avg_val_loss = validate(model, test_loader, device, val_loss_history)

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

    plot_recon_loss(recon_loss_history, start_epoch == 1, version=5)
    plot_loss(train_loss_history, val_loss_history, version=5)
    visualize_latent_space(model, test_loader, device, version=5)
    visualize_reconstructed_digits(model, device, latent_dim, version=5)
    visualize_generated_images(model, num_samples_per_class=10, device=device, noise_scale=0.1, version=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--use_cosine_distance", action="store_true",
                        help="Use cosine distance for vector quantization")
    parser.add_argument("--beta", type=float, default=BETA, help="Beta parameter for soft quantization")
    args = parser.parse_args()
    main(args)
