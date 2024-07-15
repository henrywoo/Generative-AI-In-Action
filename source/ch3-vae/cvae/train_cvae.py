import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
from hiq.cv_torch import get_cv_dataset, DS_PATH_MNIST
from hiq import deterministic

# Parameters
batch_size = 30000
original_dim = 784
latent_dim = 3
intermediate_dim = 256
epochs = 30
num_classes = 10
checkpoint_dir = 'mbin'


def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    loader_params = dict(
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    dataloader = get_cv_dataset(path=str(data_path),
                                batch_size=batch_size,
                                transform=transform,
                                return_loader=True,
                                convert_rgb=False,
                                **loader_params)
    return dataloader['train'], dataloader['test']


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc2_log_var = nn.Linear(intermediate_dim, latent_dim)
        self.fc3 = nn.Linear(num_classes, latent_dim)

    def forward(self, x, y):
        h = F.relu(self.fc1(x))
        z_mean = self.fc2_mean(h)
        z_log_var = self.fc2_log_var(h)
        yh = self.fc3(y)
        return z_mean, z_log_var, yh


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, original_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x, y):
        z_mean, z_log_var, yh = self.encoder(x, y)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var, yh


def loss_function(x, x_reconstructed, z_mean, z_log_var, yh):
    xent_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - torch.pow(z_mean - yh, 2) - torch.exp(z_log_var))
    return xent_loss + kl_loss


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch, checkpoint['train_loss_history']


def train(model, train_loader, optimizer, device, start_epoch, epochs, checkpoint_path):
    train_loss_history = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)):
            data = data.view(-1, original_dim).to(device)
            labels = F.one_hot(labels, num_classes).float().to(device)

            optimizer.zero_grad()
            x_reconstructed, z_mean, z_log_var, yh = model(data, labels)
            loss = loss_function(data, x_reconstructed, z_mean, z_log_var, yh)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        print(f'Epoch {epoch}, Loss: {avg_train_loss}')

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss_history
        }, checkpoint_path)

    return train_loss_history


def visualize_training_curve(train_loss_history, save_path):
    plt.figure(figsize=(8, 4.8))
    plt.plot(train_loss_history, label='Train Loss', marker='o', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


def visualize_latent_space(model, test_loader, device):
    from mpl_toolkits.mplot3d import Axes3D

    model.eval()
    with torch.no_grad():
        z_means = []
        labels = []
        for data, label in test_loader:
            data = data.view(-1, original_dim).to(device)
            label = label.to(device)
            z_mean, _, _ = model.encoder(data, F.one_hot(label, num_classes).float().to(device))
            z_means.append(z_mean)
            labels.append(label)
        z_means = torch.cat(z_means).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

    fig = plt.figure(figsize=(9, 9))  # Larger figure size
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(z_means[:, 0], z_means[:, 1], z_means[:, 2], c=labels, cmap='viridis')
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_zlabel('Z3')
    cbar = fig.colorbar(scatter)
    cbar.ax.tick_params(labelsize=10)  # Smaller color bar labels
    plt.savefig("latent_space.png")
    plt.show()



def generate_digits(model, device, output_digit=9, n=15, digit_size=28):
    figure = np.zeros((digit_size * n, digit_size * n))

    with torch.no_grad():
        yh = model.encoder.fc3(torch.eye(num_classes)[output_digit].to(device)).cpu().numpy()
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + yh[1]
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + yh[0]

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi, yh[2]]], device=device).float()
                x_decoded = model.decoder(z_sample).cpu().numpy()
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig("generated_digits_cvae.png")
    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = CVAE().to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    start_epoch = 1

    checkpoint_path = os.path.join(args.checkpoint_dir, 'cvae_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, train_loss_history = load_checkpoint(checkpoint_path, cvae, optimizer)
        print(f'Checkpoint loaded, resuming training from epoch {start_epoch}')
    else:
        train_loss_history = []

    train_loader, test_loader = load_data(DS_PATH_MNIST, args.batch_size)

    train_loss_history += train(cvae, train_loader, optimizer, device, start_epoch, args.epochs, checkpoint_path)

    visualize_training_curve(train_loss_history, "training_curve.png")
    visualize_latent_space(cvae, test_loader, device)
    generate_digits(cvae, device, output_digit=9, n=15, digit_size=28)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAE Training Script")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=epochs, help="Number of epochs to train")
    parser.add_argument("--checkpoint_dir", type=str, default=checkpoint_dir, help="Directory to save checkpoints")
    args = parser.parse_args()
    main(args)

"""
python train_cvae.py --batch_size 30000 --epochs 30 --checkpoint_dir mbin
"""