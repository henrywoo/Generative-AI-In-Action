import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
import shutil
from model import VariationalAutoencoder

# --- Data Handling ---
def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])  # Convert to PyTorch Tensors
    train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    valid_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

# --- Loss Function ---
def vae_loss(reconstruction, original, mean, log_var):
    reconstruction_loss = F.mse_loss(reconstruction, original) #, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss, kl_divergence


def train(model, train_loader, optimizer, device, beta):
    model.train()
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, log_var = model(data)
        recon_batch = recon_batch.view(-1, 28 * 28)  # Flatten the output
        data = data.view(-1, 28 * 28)  # Flatten the input
        recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        loss = recon_loss + beta * kl_loss
        loss.backward()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
        optimizer.step()
    train_recon_loss /= len(train_loader.dataset)
    train_kl_loss /= len(train_loader.dataset)
    return train_recon_loss, train_kl_loss

def validate(model, valid_loader, device, beta):
    model.eval()
    valid_recon_loss = 0
    valid_kl_loss = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(device)
            recon_batch, mean, log_var = model(data)
            recon_batch = recon_batch.view(-1, 28 * 28)  # Flatten the output
            data = data.view(-1, 28 * 28)  # Flatten the input
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            valid_recon_loss += recon_loss.item()
            valid_kl_loss += kl_loss.item()
    valid_recon_loss /= len(valid_loader.dataset)
    valid_kl_loss /= len(valid_loader.dataset)
    return valid_recon_loss, valid_kl_loss

# Define the model, training, and validation functions here...
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(state, is_best, filename='vae_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'vae_best.pth.tar')


def main(args):
    DATA_PATH = Path(args.data_path)
    MODEL_PATH = Path(args.model_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = load_data(DATA_PATH, args.batch_size)
    model = VariationalAutoencoder(args.codings_size).to(device)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume and (MODEL_PATH / "vae_fashion_mnist.pth").exists():
        checkpoint = torch.load(MODEL_PATH / "vae_fashion_mnist.pth")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}')

    torch.manual_seed(args.seed)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    train_recon_losses = []
    train_kl_losses = []
    valid_recon_losses = []
    valid_kl_losses = []

    for epoch in range(start_epoch, args.epochs + 1):
        beta = min(1.0, epoch / args.kl_annealing_epochs)
        train_recon_loss, train_kl_loss = train(model, train_loader, optimizer, device, beta)
        valid_recon_loss, valid_kl_loss = validate(model, valid_loader, device, beta)

        train_recon_losses.append(train_recon_loss)
        train_kl_losses.append(train_kl_loss)
        valid_recon_losses.append(valid_recon_loss)
        valid_kl_losses.append(valid_kl_loss)

        print(
            f'Epoch: {epoch} Train Recon Loss: {train_recon_loss:.4f}, Train KL Loss: {train_kl_loss:.4f}, Valid Recon Loss: {valid_recon_loss:.4f}, Valid KL Loss: {valid_kl_loss:.4f}')

        early_stopping(valid_recon_loss + valid_kl_loss)
        if early_stopping.early_stop:
            print('Early stopping')
            break

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'codings_size': args.codings_size,
        }, is_best=False, filename=MODEL_PATH / "vae_fashion_mnist.pth")

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 4))
    plt.plot(train_recon_losses, label='Train Recon Loss', marker='o', alpha=0.5)
    plt.plot(train_kl_losses, label='Train KL Loss', marker='v', alpha=0.5)
    plt.plot(valid_recon_losses, label='Valid Recon Loss', marker='x', alpha=0.5)
    plt.plot(valid_kl_losses, label='Valid KL Loss', marker='*', alpha=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_PATH / "vae_fashion_mnist_losses.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder for FashionMNIST")
    parser.add_argument('--data_path', type=str, default='data', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='models', help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--codings_size', type=int, default=10, help='Size of the latent codings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer (e.g., Adam, SGD, etc.)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from')
    parser.add_argument('--kl_annealing_epochs', type=int, default=10, help='Epochs over which to anneal KL term')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
