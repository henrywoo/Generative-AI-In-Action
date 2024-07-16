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

# 基本参数
original_dim = 784
latent_dim = 3
intermediate_dim = 256
kappa = 20
epsilon = 1e-7

def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
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

class SVAE(nn.Module):
    def __init__(self):
        super(SVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x ** 2) * (latent_dim - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        self.W = torch.tensor(np.interp(np.random.random(10 ** 6), y, x), dtype=torch.float32)

    def encode(self, x):
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
        mu = self.encode(x.view(-1, original_dim))
        z = self.reparameterize(mu)
        return self.decode(z), mu

    def generate(self, num_samples, device, noise_scale=0.05):
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

def loss_function(recon_x, x, vq_loss=0):
    recon_loss = F.mse_loss(recon_x, x.view(-1, original_dim), reduction='sum')
    return recon_loss + vq_loss, recon_loss


def plot_recon_loss(recon_loss_history, version):
    epochs = range(1, len(recon_loss_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, recon_loss_history, label='Reconstruction Loss', marker='o', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Recon Loss vs. Epochs (v{version})')
    plt.legend()
    plt.grid(True)
    filename = f"vq_svae_v{version}_loss_recon.png"
    plt.savefig(filename)
    plt.show()

    # Save recon_loss_history to recon_loss.csv
    save_recon_loss_history(recon_loss_history, version)


def save_recon_loss_history(recon_loss_history, version):
    if len(recon_loss_history)==0:
        return
    import pandas as pd
    df = pd.DataFrame(recon_loss_history, columns=[f'v{version}'])
    filename = 'recon_loss.csv'
    try:
        existing_df = pd.read_csv(filename)
        existing_df[f'v{version}'] = recon_loss_history
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

def train(model, epoch, train_loader, optimizer, device, train_loss_history, recon_loss_history):
    model.train()
    train_loss = 0
    total_recon_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)):
        data = data.view(-1, original_dim).to(device)
        optimizer.zero_grad()
        recon_batch, mu = model(data)
        loss, recon_loss = loss_function(recon_batch, data, 0)
        loss.backward()
        train_loss += loss.item()
        total_recon_loss += recon_loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    train_loss_history.append(avg_train_loss)
    recon_loss_history.append(avg_recon_loss)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')
    print(f'====> Epoch: {epoch} Average recon loss: {avg_recon_loss:.4f}')

def validate(model, test_loader, device, val_loss_history):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, original_dim).to(device)
            recon_batch, mu = model(data)
            t, _ = loss_function(recon_batch, data, 0)
            test_loss += t.item()
    avg_val_loss = test_loss / len(test_loader.dataset)
    val_loss_history.append(avg_val_loss)
    print(f'====> Test set loss: {avg_val_loss:.4f}')


def visualize_latent_space(model, test_loader, device, version=0):
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
        z_means = torch.cat(z_means).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

    fig = plt.figure(figsize=(15, 15))  # Larger figure size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_means[:, 0], z_means[:, 1], z_means[:, 2], c=labels, cmap='tab10', s=35)  # Smaller points

    # Create legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)

    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')
    plt.title('Latent Space Visualization')
    plt.savefig(f"latent_space_v{version}.png")
    plt.show()


def visualize_reconstructed_digits(model, device, latent_dim, version=0):
    with torch.no_grad():
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        for i in range(n):
            for j in range(n):
                z_sample = torch.randn(1, latent_dim, device=device)
                z_sample /= z_sample.norm()
                x_decoded = model.decode(z_sample).view(digit_size, digit_size).cpu()
                digit = x_decoded.numpy()
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.title('Reconstructed Digits')
    plt.savefig(f'reconstructed_digits_v{version}.png')
    plt.show()

def plot_loss(train_loss_history, val_loss_history, version=0):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss', marker='o', alpha=0.5)
    plt.plot(val_loss_history, label='Validation Loss', marker='o', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"vq_svae_v{version}_loss_history.png")
    plt.show()

def visualize_generated_images(model, num_samples, device, noise_scale=0.1, version=0):
    model.eval()
    with torch.no_grad():
        samples = model.generate(num_samples, device, noise_scale)
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
        for i in range(num_samples):
            axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"generated_images_vq_svae_v{version}.png")
        plt.show()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history = []
    recon_loss_history = []
    val_loss_history = []
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'svae_checkpoint_v0.pth')
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

    plot_recon_loss(recon_loss_history, 0)
    plot_loss(train_loss_history, val_loss_history)
    visualize_latent_space(model, test_loader, device)
    visualize_reconstructed_digits(model, device, latent_dim)
    visualize_generated_images(model, num_samples=10, device=device, noise_scale=0.1, version=0)

BATCH_SIZE = 64
EPOCHS = 50
LR = 5e-4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVAE Training Script")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default='mbin', help="Directory to save checkpoints")
    args = parser.parse_args()
    main(args)
