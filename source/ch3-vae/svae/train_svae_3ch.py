import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hiq.cv_torch import get_cv_dataset, DS_PATH_MNIST
from hiq import deterministic
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

# 基本参数
original_dim = 784
latent_dim = 3
intermediate_dim = 256
kappa = 20
epsilon = 1e-7
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-4
BETA = 10.0
BOOK_SIZE = 2048
COMMITMENT_COST = 0.25
VQ_LOSS_WEIGHT = 50
PATIENCE = 20
CONTRASTIVE = False
vMFLoss = True
vMFLossWeight = 1.0
CONTR_MUL = 0.3
CNN_NETWORK = False
MARGIN = 0.08
T_MAX = EPOCHS
ETA_MIN = LR / 100


def load_data(data_path, batch_size):
    channel = 1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * channel, (0.5,) * channel)])
    loader_params = dict(
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    dataloader = get_cv_dataset(path=str(data_path),
                                batch_size=batch_size,
                                num_workers=8,
                                transform=transform,
                                image_size=None,
                                return_type="pair",
                                return_loader=True,
                                convert_rgb=False,
                                **loader_params)
    return dataloader['train'], dataloader['test']


class SVAE_CNN(nn.Module):
    def __init__(self, embedding_dim=84):
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
            nn.Tanh()
        )
        self.mode = 'cnn'


class SVAE_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Tanh()
        )
        self.mode = 'linear'


class SVAE(SVAE_CNN):
    def __init__(self):
        super().__init__()
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x ** 2) * (latent_dim - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        self.W = torch.tensor(np.interp(np.random.random(10 ** 6), y, x), dtype=torch.float32)

    def encode(self, x):
        if self.mode == 'cnn':
            x = x.view(-1, 1, 28, 28)
        h = self.encoder(x)
        mu = F.normalize(h, p=2, dim=-1)
        return mu

    def reparameterize(self, mu):
        dims = mu.size(-1)
        idx = torch.randint(0, 10 ** 6, (mu.size(0), 1), dtype=torch.long, device=mu.device)
        w = self.W.to(mu.device)[idx]
        eps = torch.randn_like(mu)
        nu = eps - (eps * mu).sum(dim=1, keepdim=True) * mu
        nu = F.normalize(nu, p=2, dim=-1)
        return w * mu + torch.sqrt(1 - w ** 2) * nu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """return recon_batch, vq_loss, quantized"""
        mu = self.encode(x.view(-1, original_dim))
        z = self.reparameterize(mu)
        r = self.decode(z)
        if self.mode == 'cnn':
            r = r.view(r.shape[0], -1)
        return r, None, mu

    def generate(self, num_samples, device, noise_scale=0.05, class_means=None):
        if class_means is None:
            return self.generate_rand(num_samples, device, noise_scale)
        else:
            return self.generate_with_guidance(num_samples, device, noise_scale, class_means)

    def generate_rand(self, num_samples, device, noise_scale=0.05):
        with torch.no_grad():
            dims = latent_dim
            idx = torch.randint(0, 10 ** 6, (num_samples, 1), dtype=torch.long, device=device)
            w = self.W.to(device)[idx]
            mu = torch.randn(num_samples, latent_dim, device=device)
            mu = F.normalize(mu, p=2, dim=-1)
            eps = torch.randn_like(mu)
            nu = eps - (eps * mu).sum(dim=1, keepdim=True) * mu
            nu = F.normalize(nu, p=2, dim=-1)
            z = w * mu + torch.sqrt(1 - w ** 2) * nu
            z += noise_scale * torch.randn_like(z)
            samples = self.decode(z).cpu()
            return samples

    def generate_with_guidance(self, num_samples, device, noise_scale, class_means):
        with torch.inference_mode():
            dims = latent_dim
            selected_means = class_means[torch.randint(0, class_means.shape[0], (num_samples,))]
            noise = torch.randn(num_samples, dims, device=device) * noise_scale
            z = selected_means.to(device) + noise
            samples = self.decode(z).cpu()
            return samples

def loss_function(recon_x, x, vq_loss=0):
    recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='mean')
    return recon_loss + none_as_0(vq_loss), recon_loss


def plot_recon_loss(recon_loss_history, replace_it, version):
    if not recon_loss_history:
        return
    epochs = range(1, len(recon_loss_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, recon_loss_history, label='Reconstruction Loss', marker='o', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Recon Loss vs. Epochs (v{version})')
    plt.legend()
    plt.grid(True)
    filename = f"img/vq_svae_v{version}_loss_recon.png"
    plt.savefig(filename)
    plt.show()

    # Save recon_loss_history to recon_loss.csv
    try:
        save_recon_loss_history(recon_loss_history, replace_it, version)
    except:
        pass


def save_recon_loss_history(recon_loss_history, replace_it, version):
    import pandas as pd
    if len(recon_loss_history) == 0:
        return

    df = pd.DataFrame(recon_loss_history, columns=[f'v{version}'])
    filename = 'recon_loss.csv'

    try:
        existing_df = pd.read_csv(filename)

        if f'v{version}' in existing_df.columns:
            existing_data = existing_df[f'v{version}'].dropna().tolist()
            if len(existing_data) == len(recon_loss_history) or replace_it:
                # Replace old data with new data
                existing_df[f'v{version}'] = pd.Series(recon_loss_history)
            else:
                # Append new data to existing column
                updated_data = existing_data + recon_loss_history
                existing_df[f'v{version}'] = pd.Series(updated_data)
        else:
            # Add new column for current version
            existing_df[f'v{version}'] = pd.Series(recon_loss_history)

        # Ensure all columns have the same length by padding with NaN
        max_length = max(existing_df.shape[0], len(recon_loss_history))
        for column in existing_df.columns:
            if len(existing_df[column]) < max_length:
                existing_df[column] = existing_df[column].tolist() + [float('nan')] * (
                        max_length - len(existing_df[column]))

        # Ensure the new column also matches the max length
        if len(existing_df[f'v{version}']) < max_length:
            existing_df[f'v{version}'] = existing_df[f'v{version}'].tolist() + [float('nan')] * (
                    max_length - len(existing_df[f'v{version}']))

        existing_df.to_csv(filename, index=False)
    except FileNotFoundError:
        df.to_csv(filename, index=False)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    train_loss_history = checkpoint['train_loss_history']
    val_loss_history = checkpoint['val_loss_history']
    return start_epoch, train_loss_history, val_loss_history


def contrastive_loss_cosine(z, labels, margin=0.05, multiplier=500.):
    z = F.normalize(z, p=2, dim=1)
    sim_matrix = torch.matmul(z, z.t())
    label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    positive_loss = label_matrix * (1 - sim_matrix)
    negative_loss = (1 - label_matrix) * F.relu(sim_matrix - margin)
    contrastive_loss_value = multiplier * (positive_loss + negative_loss).mean()
    return contrastive_loss_value


def none_as_0(x):
    if isinstance(x, int):
        return x
    else:
        return 0 if x is None else x.item()


def vmfml_loss(z, labels, kappa=100, num_classes=10):
    z = F.normalize(z, p=2, dim=1)
    batch_size = z.size(0)
    # Compute class means
    class_means = torch.zeros(num_classes, z.size(1), device=z.device)
    for c in range(num_classes):
        class_indices = (labels == c)
        if class_indices.sum() > 0:
            class_means[c] = F.normalize(z[class_indices].mean(dim=0), p=2, dim=0)
    # Compute the loss
    loss = 0
    for i in range(batch_size):
        label = labels[i]
        mean = class_means[label]
        loss += -kappa * torch.dot(z[i], mean)
    loss = loss / batch_size
    return loss * 0.05


def compute_class_means(z, labels, num_classes):
    class_means = torch.zeros(num_classes, z.shape[1])
    for c in range(num_classes):
        class_indices = (labels == c)
        if class_indices.sum() > 0:
            class_means[c] = F.normalize(z[class_indices].mean(dim=0), p=2, dim=0)
    return class_means


def vmfml_loss_v2(z, labels, kappa=100, num_classes=10):
    z = F.normalize(z, p=2, dim=1)
    batch_size = z.size(0)
    # Compute class means
    class_means = torch.zeros(num_classes, z.size(1), device=z.device)
    for c in range(num_classes):
        class_indices = (labels == c)
        if class_indices.sum() > 0:
            class_means[c] = F.normalize(z[class_indices].mean(dim=0), p=2, dim=0)
    # Compute posterior probabilities
    logits = kappa * torch.mm(z, class_means.t())
    log_probs = F.log_softmax(logits, dim=1)
    # Create one-hot encoding of labels
    labels_one_hot = F.one_hot(labels, num_classes).float()
    # Compute the vMFML loss as the negative log likelihood
    loss = -torch.sum(labels_one_hot * log_probs) / batch_size
    return loss


def train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history, vq_loss_weight=1,
          use_vmfml=False, scheduler=None, original_dim=784, kappa=10, num_classes=10, contrastive=False):
    model.train()
    train_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0  # Initialize total VQ loss
    total_vmfml_loss = 0  # Initialize total vMFML loss
    total_contrastive_loss = 0  # Initialize total contrastive loss
    contrastive_loss_value, vmfml_loss_value, kld = [torch.tensor(0.0, device=device)] * 3
    batch_idx = 1
    small_constant = 1e-8
    with tqdm(total=len(train_loader.dataset), desc=f"Train Epoch {epoch}", unit='samples') as pbar:
        for batch_idx, (images, labels) in enumerate(train_loader):
            data = images.to(device).view(-1, original_dim)
            optimizer.zero_grad()
            recon_batch, vq_loss, quantized = model(data)
            if vq_loss is not None:
                vq_loss *= vq_loss_weight
            if contrastive:
                z = quantized.view(quantized.size(0), -1)
                contrastive_loss_value = contrastive_loss_cosine(z, labels.to(device), multiplier=CONTR_MUL)
            if use_vmfml:
                z = quantized.view(quantized.size(0), -1)
                vmfml_loss_value = vmfml_loss_v2(z, labels.to(device), kappa=kappa, num_classes=num_classes)

            loss, recon_loss = loss_function(recon_batch, data, vq_loss)
            loss += vmfml_loss_value * vMFLossWeight + contrastive_loss_value + kld
            loss.backward()
            train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += none_as_0(vq_loss)  # Accumulate VQ loss
            total_vmfml_loss += vmfml_loss_value.item()  # Accumulate vMFML loss
            total_contrastive_loss += contrastive_loss_value.item()  # Accumulate contrastive loss
            optimizer.step()
            if scheduler:
                scheduler.step()
            pbar.update(data.size(0))
            pbar.set_postfix({'Recon': recon_loss.item(),
                              'VQ': none_as_0(vq_loss),
                              'Contra': contrastive_loss_value.item(),
                              'vMFML': vmfml_loss_value.item(),
                              'kld': kld.item()})

    avg_train_loss = train_loss / batch_idx
    avg_recon_loss = total_recon_loss / batch_idx
    avg_vq_loss = total_vq_loss / batch_idx  # Compute average VQ loss
    avg_vmfml_loss = total_vmfml_loss / batch_idx
    avg_contrastive_loss = total_contrastive_loss / batch_idx
    train_loss_history.append(avg_train_loss)
    recon_loss_history.append(avg_recon_loss)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f},'
          f' recon: {avg_recon_loss:.4f}, vq: {avg_vq_loss:.4f},'
          f' contra: {avg_contrastive_loss:.4f}, vMFML: {avg_vmfml_loss:.4f}')

    # Log metrics to wandb
    wandb.log({"t/avg_train_loss": avg_train_loss,
               "t/avg_recon_loss": avg_recon_loss,
               "t/avg_vq_loss": avg_vq_loss,
               "t/avg_contrastive_loss": avg_contrastive_loss,
               "t/avg_vmfml_loss": avg_vmfml_loss,
               "t/lr": scheduler.get_last_lr()[0] if scheduler else None})

    # Log images to wandb
    n = 8  # Number of images to log
    original_images = data[:n].view(-1, 1, 28, 28).cpu()
    reconstructed_images = recon_batch[:n].view(-1, 1, 28, 28).cpu()
    comparison = torch.cat([original_images, reconstructed_images])
    grid = make_grid(comparison, nrow=n)
    wandb.log({"Reconstructions": [wandb.Image(grid, caption="Top: Original images, Bottom: Reconstructed images")]})


def validate(model, test_loader, device, val_loss_history, enable_statistics=False, vq_loss_weight=1, contrastive=False,
             use_vmfml=False, kappa=10, num_classes=10):
    model.eval()
    test_loss = 0
    total_recon_loss = 0
    total_vmfml_loss = 0  # Initialize total vMFML loss
    total_contrastive_loss = 0  # Initialize total contrastive loss
    contrastive_loss_value = torch.tensor(0.0, device=device)
    vmfml_loss_value = torch.tensor(0.0, device=device)

    with torch.no_grad():
        if enable_statistics:
            model.vq_layer.enable_statistics = True
        batch_idx = 1
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.view(-1, original_dim).to(device)
            recon_batch, vq_loss, quantized = model(data)
            if vq_loss is not None:
                vq_loss *= vq_loss_weight
            if contrastive:
                z = quantized.view(quantized.size(0), -1)
                contrastive_loss_value = contrastive_loss_cosine(z, labels.to(device), multiplier=CONTR_MUL)
            if use_vmfml:
                z = quantized.view(quantized.size(0), -1)
                vmfml_loss_value = vmfml_loss_v2(z, labels.to(device), kappa=kappa, num_classes=num_classes)

            t, recon_loss = loss_function(recon_batch, data, vq_loss)
            t += contrastive_loss_value + vmfml_loss_value * vMFLossWeight
            test_loss += t.item()
            total_recon_loss += recon_loss.item()
            total_vmfml_loss += vmfml_loss_value.item()  # Accumulate vMFML loss
            total_contrastive_loss += contrastive_loss_value.item()  # Accumulate contrastive loss

    avg_val_loss = test_loss / batch_idx
    val_loss_history.append(avg_val_loss)
    if enable_statistics:
        print(model.vq_layer.get_codebook_statistics())
        model.vq_layer.reset_statistics()
        model.vq_layer.enable_statistics = False
    print(f'====> Total Test loss: {avg_val_loss:.4f}')

    wandb.log({"v/avg_val_loss": avg_val_loss,
               "v/avg_recon_loss": total_recon_loss / batch_idx,
               "v/avg_vmfml_loss": total_vmfml_loss / batch_idx,
               "v/avg_contrastive_loss": total_contrastive_loss / batch_idx})

    return avg_val_loss


def visualize_latent_space(model, test_loader, device, num_classes=NUM_CLASSES, version=0):
    if latent_dim != 3:
        return
    model.eval()
    with torch.no_grad():
        z_means = []
        labels = []
        for data, label in test_loader:
            data = data.view(-1, original_dim).to(device)
            label = label.to(device)
            z_mean = model.encode(data)
            if hasattr(model, 'quant'):
                _, z_mean = model.quant(z_mean)
                z_mean = z_mean.view(z_mean.shape[0], latent_dim)
            z_means.append(z_mean)
            labels.append(label)
        z_means = torch.cat(z_means)
        labels = torch.cat(labels)
        z_means_cpu = z_means.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        class_means = compute_class_means(z_means, labels, num_classes).to(device)
    fig = plt.figure(figsize=(15, 15))  # Larger figure size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_means_cpu[:, 0], z_means_cpu[:, 1], z_means_cpu[:, 2], c=labels_cpu, cmap='tab10',
                         s=5)  # Smaller points
    # Create legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')
    plt.title('Latent Space Visualization')
    plt.savefig(f"img/latent_space_v{version}.png")
    plt.show()

    return class_means


def visualize_reconstructed_digits(model, device, latent_dim, version=0, class_means=None):
    import random
    with torch.no_grad():
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        for i in range(n):
            for j in range(n):
                # TODO - 采样的时候去中心点采样
                if class_means is None:
                    z_sample = torch.randn(1, latent_dim, device=device)
                else:
                    z_sample = torch.randn(1, latent_dim, device=device) / 10 + class_means[random.randint(0, 9)]
                z_sample /= z_sample.norm()
                x_decoded = model.decode(z_sample).view(digit_size, digit_size).cpu()
                digit = x_decoded.numpy()
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Reconstructed Digits')
    plt.savefig(f'img/reconstructed_digits_v{version}.png')
    plt.show()


def plot_loss(train_loss_history, val_loss_history, version=0):
    if not train_loss_history:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss', marker='o', alpha=0.5)
    plt.plot(val_loss_history, label='Validation Loss', marker='o', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"img/vq_svae_v{version}_loss_history.png")
    plt.show()


def visualize_generated_images(model, num_samples, device, noise_scale=0.1, class_means=None, version=0):
    model.eval()
    with torch.no_grad():
        samples = model.generate(num_samples, device, noise_scale, class_means)
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
        for i in range(num_samples):
            axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"img/generated_images_vq_svae_v{version}.png")
        plt.show()


def generate_spherical_points(dim, num_points):
    if dim != 3:
        raise ValueError("This function currently only supports 3-dimensional points")

    points = []
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for i in range(num_points):
        z = 1 - (i / float(num_points - 1)) * 2  # z goes from 1 to -1
        radius = np.sqrt(1 - z * z)  # radius at z
        theta = 2 * np.pi * i / phi
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
        points.append([x, y, z])
    points = np.array(points)
    points = torch.tensor(points, dtype=torch.float)
    return points


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


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)
    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1
    checkpoint_path = os.path.join(args.checkpoint_dir, 'svae_checkpoint_v0.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history, val_loss_history = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')

    if args.mode == "train":
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = PATIENCE
        for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
            train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history,
                  contrastive=CONTRASTIVE,
                  use_vmfml=vMFLoss,
                  scheduler=scheduler)
            avg_val_loss = validate(model, test_loader, device, val_loss_history,
                                    contrastive=CONTRASTIVE,
                                    use_vmfml=vMFLoss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history
            }, checkpoint_path)

        plot_recon_loss(recon_loss_history, start_epoch == 1, version=0)
        plot_loss(train_loss_history, val_loss_history)
    class_means = visualize_latent_space(model, test_loader, device, version=0)
    visualize_reconstructed_digits(model, device, latent_dim, class_means=class_means, version=0)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1,
                               class_means=class_means, version=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVAE Training Script")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    parser.add_argument("--mode", type=str, default='train', help="train or infer")
    args = parser.parse_args()
    if args.mode == 'train':
        wandb.init(
            project="svae_mnist",
            name="vmfloss-beta-sVAE",
            config={
                "learning_rate": LR,
            }
        )
    main(args)
