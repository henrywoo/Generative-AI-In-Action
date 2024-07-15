# This is actually a beta-VAE.
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
import shutil
from model import VariationalAutoencoder
from hiq.cv_torch import get_cv_dataset, DS_PATH_FASHION_MNIST


# --- Data Handling ---
def load_data(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    loader_params = dict(
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    dataloader = get_cv_dataset(path=str(data_path),
                                batch_size=batch_size,
                                num_workers=2,
                                transform=transform,
                                image_size=None,
                                return_type="pair",
                                return_loader=True,
                                convert_rgb=False,
                                **loader_params)
    return dataloader['train'], dataloader['test']


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


def save_checkpoint(state, is_best, folder, filename):
    torch.save(state, f'{folder}/{filename}')
    if is_best:
        shutil.copyfile(filename, f'{folder}/best.pt')


def main(args):
    DATA_PATH = Path(args.data_path)
    MODEL_PATH = Path(args.model_path)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    train_loader, valid_loader = load_data(DATA_PATH, args.batch_size)
    model = VariationalAutoencoder(args.codings_size).to(device)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume and (MODEL_PATH / "best.pt").exists():
        checkpoint = torch.load(MODEL_PATH / "best.pt")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}')

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
        }, is_best=False, folder=MODEL_PATH, filename=f"vae_{epoch}.pt")

    plt.style.use('ggplot')
    plt.figure(figsize=(16, 4))
    plt.plot(train_recon_losses, label='Train Recon Loss', marker='o', alpha=0.5)
    plt.plot(train_kl_losses, label='Train KL Loss', marker='v', alpha=0.5)
    plt.plot(valid_recon_losses, label='Valid Recon Loss', marker='x', alpha=0.5)
    plt.plot(valid_kl_losses, label='Valid KL Loss', marker='*', alpha=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/vae_fashion_mnist_losses.png")
    plt.show()


if __name__ == "__main__":
    from hiq import deterministic
    parser = argparse.ArgumentParser(description="Variational Autoencoder for FashionMNIST")
    # fashion_mnist
    parser.add_argument('--data_path', type=str, default=DS_PATH_FASHION_MNIST, help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='mbin', help='Path to save the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--codings_size', type=int, default=10, help='Size of the latent codings')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer (e.g., Adam, SGD, etc.)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to start training from')
    parser.add_argument('--kl_annealing_epochs', type=int, default=10, help='Epochs over which to anneal KL term')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
