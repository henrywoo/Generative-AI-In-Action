import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import os
from hiq import print_model, deterministic


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAN on the FashionMNIST dataset")
    parser.add_argument('--img_size', type=int, default=28, help='Size of each image dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--z_size', type=int, default=100, help='Size of the latent vector')
    parser.add_argument('--g_layers', nargs='+', type=int, default=[256, 512, 1024],
                        help='Sizes of the generator layers')
    parser.add_argument('--d_layers', nargs='+', type=int, default=[1024, 512, 256],
                        help='Sizes of the discriminator layers')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for the dataset')
    parser.add_argument('--output_dir', type=str, default='img', help='Directory for saving output images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, default="checkpoints/checkpoint_29.pth",
                        help='Path to checkpoint to resume training')
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.label_emb = nn.Embedding(class_num, class_num)
        self.model = nn.Sequential(
            nn.Linear(z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        z = z.view(-1, self.z_size)
        if labels is not None:
            c = self.label_emb(labels.to(z.device))  # Ensure the labels are on the same device as z
        else:
            c = torch.zeros(z.size(0), self.class_num).to(z.device)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size):
        super().__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(self.img_size * self.img_size, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.img_size * self.img_size)
        out = self.model(x)
        return out.squeeze()


def initialize_dataloader(data_dir, batch_size, img_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth'))


def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    return checkpoint['epoch'] + 1


def train(generator, discriminator, data_loader, g_optimizer, d_optimizer, criterion, epochs, z_size, class_num, device,
          output_dir, checkpoint_dir, resume, guidance_scale=0.5):
    start_epoch = 0
    if resume and os.path.exists(resume):
        start_epoch = load_checkpoint(resume, generator, discriminator, g_optimizer, d_optimizer)
        if start_epoch < epochs:
            print(f"Resuming training from epoch {start_epoch + 1}...")

    for epoch in range(start_epoch, epochs):
        print(f'Starting epoch {epoch + 1}...')
        for images, labels in data_loader:
            real_images = Variable(images).to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            d_optimizer.zero_grad()
            real_validity = discriminator(real_images)
            real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

            z = Variable(torch.randn(batch_size, z_size)).to(device)
            fake_images_cond = generator(z, labels)
            fake_images_uncond = generator(z)

            fake_validity_cond = discriminator(fake_images_cond.detach())
            fake_validity_uncond = discriminator(fake_images_uncond.detach())

            fake_loss_cond = criterion(fake_validity_cond, Variable(torch.zeros(batch_size)).to(device))
            fake_loss_uncond = criterion(fake_validity_uncond, Variable(torch.zeros(batch_size)).to(device))

            d_loss = real_loss + guidance_scale * fake_loss_cond + (1 - guidance_scale) * fake_loss_uncond
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            fake_images_cond = generator(z, labels)
            fake_images_uncond = generator(z)

            fake_validity_cond = discriminator(fake_images_cond)
            fake_validity_uncond = discriminator(fake_images_uncond)

            g_loss_cond = criterion(fake_validity_cond, Variable(torch.ones(batch_size)).to(device))
            g_loss_uncond = criterion(fake_validity_uncond, Variable(torch.ones(batch_size)).to(device))

            g_loss = guidance_scale * g_loss_cond + (1 - guidance_scale) * g_loss_uncond
            g_loss.backward()
            g_optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')

        if (epoch + 1) % 10 == 0:
            z = Variable(torch.randn(class_num, z_size)).to(device)
            labels = Variable(torch.arange(class_num)).to(device)
            sample_images = generator(z, labels).unsqueeze(1).data.cpu()
            grid = make_grid(sample_images, nrow=class_num, normalize=True).permute(1, 2, 0).numpy()
            plt.imshow(grid)
            plt.savefig(os.path.join(output_dir, f"{epoch + 1}.png"))
            plt.show()

        save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_dir)


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, z_size, class_num, device):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels, z_size,
                             class_num, device):
    d_optimizer.zero_grad()
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, class_num, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def show_final_generated_images(generator, z_size, class_num, class_list, device, output_dir):
    num_images = class_num * (class_num // 2)
    z = Variable(torch.randn(num_images, z_size)).to(device)
    labels = Variable(torch.LongTensor([i for _ in range(class_num // 2) for i in range(class_num)])).to(device)
    sample_images = generator(z, labels).unsqueeze(1).data.cpu()
    grid = make_grid(sample_images, nrow=class_num, normalize=True).permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(5.8, 0.72 * (class_num // 2)))
    ax.imshow(grid)
    _ = plt.yticks([])
    _ = plt.xticks(np.arange(15, 300, 30), class_list, rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final.png"))
    plt.show()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_list = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_num = len(class_list)
    train_loader = initialize_dataloader(args.data_dir, args.batch_size, args.img_size)
    generator = Generator(args.g_layers, args.z_size, args.img_size, class_num).to(device)
    discriminator = Discriminator(args.d_layers, args.img_size).to(device)
    print_model(generator, legend=True)
    print_model(generator)

    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    train(generator, discriminator, train_loader, g_optimizer, d_optimizer, criterion, args.epochs, args.z_size,
          class_num, device, args.output_dir, args.checkpoint_dir, args.resume, guidance_scale=0.5)

    show_final_generated_images(generator, args.z_size, class_num, class_list, device, args.output_dir)


if __name__ == "__main__":
    main()
