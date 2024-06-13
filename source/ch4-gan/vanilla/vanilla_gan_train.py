import torch
import os
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.optim import Adam
from dataset_mnist import MnistDataset
from torch.utils.data import DataLoader
from hiq.vis import print_model
from hiq import set_seed
import cv2  # Import OpenCV
import argparse

set_seed(has_torch=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_video_writer(frame_size, output_file='gan_training.mp4', fps=2.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    return video_writer


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        activation = nn.LeakyReLU()
        layers_dim = [self.latent_dim, 128, 256, 512, self.img_size[0] * self.img_size[1] * self.channels]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.BatchNorm1d(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Tanh()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, z):
        batch_size = z.shape[0]
        out = z.reshape(-1, self.latent_dim)
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(batch_size, self.channels, self.img_size[0], self.img_size[1])
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super().__init__()
        self.img_size = img_size
        self.channels = channels
        activation = nn.LeakyReLU()
        layers_dim = [self.img_size[0] * self.img_size[1] * self.channels, 512, 256, 128, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers_dim[i], layers_dim[i + 1]),
                nn.LayerNorm(layers_dim[i + 1]) if i != len(layers_dim) - 2 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity()
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        out = x.reshape(-1, self.img_size[0] * self.img_size[1] * self.channels)
        for layer in self.layers:
            out = layer(out)
        return out


def train(latent_dim, img_size, channels, im_path, im_ext, batch_size, num_epochs, num_samples, nrows, output_dir):
    mnist = MnistDataset('train', im_path=im_path, im_ext=im_ext)
    mnist_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim, img_size, channels).to(device)
    discriminator = Discriminator(img_size, channels).to(device)
    generator.train()
    discriminator.train()
    print_model(generator)
    print_model(discriminator)

    optimizer_generator = Adam(generator.parameters(), lr=1E-4, betas=(0.5, 0.999))
    optimizer_discriminator = Adam(discriminator.parameters(), lr=1E-4, betas=(0.5, 0.999))

    criterion = torch.nn.BCEWithLogitsLoss()

    steps = 0
    generated_sample_count = 0
    video_writer = None

    for epoch_idx in tqdm(range(num_epochs)):
        generator_losses = []
        discriminator_losses = []
        mean_real_dis_preds = []
        mean_fake_dis_preds = []

        for im in mnist_loader:
            real_ims = im.float().to(device)
            batch_size = real_ims.shape[0]

            optimizer_discriminator.zero_grad()
            fake_im_noise = torch.randn((batch_size, latent_dim), device=device)
            fake_ims = generator(fake_im_noise)
            real_label = torch.ones((batch_size, 1), device=device)
            fake_label = torch.zeros((batch_size, 1), device=device)

            disc_real_pred = discriminator(real_ims)
            disc_fake_pred = discriminator(fake_ims.detach())
            disc_real_loss = criterion(disc_real_pred.reshape(-1), real_label.reshape(-1))
            mean_real_dis_preds.append(torch.nn.Sigmoid()(disc_real_pred).mean().item())

            disc_fake_loss = criterion(disc_fake_pred.reshape(-1), fake_label.reshape(-1))
            mean_fake_dis_preds.append(torch.nn.Sigmoid()(disc_fake_pred).mean().item())
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_loss.backward()
            optimizer_discriminator.step()

            optimizer_generator.zero_grad()
            fake_im_noise = torch.randn((batch_size, latent_dim), device=device)
            fake_ims = generator(fake_im_noise)
            disc_fake_pred = discriminator(fake_ims)
            gen_fake_loss = criterion(disc_fake_pred.reshape(-1), real_label.reshape(-1))
            gen_fake_loss.backward()
            optimizer_generator.step()

            generator_losses.append(gen_fake_loss.item())
            discriminator_losses.append(disc_loss.item())

            if steps % 50 == 0:
                with torch.no_grad():
                    generator.eval()
                    infer(generated_sample_count, generator, latent_dim, num_samples, nrows, video_writer, output_dir)
                    generated_sample_count += 1
                    generator.train()
            steps += 1

        torch.save(generator.state_dict(), os.path.join(output_dir, 'generator_ckpt.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_ckpt.pth'))

    print('Done Training ...')
    if video_writer:
        video_writer.release()


def infer(generated_sample_count, generator, latent_dim, num_samples, nrows, video_writer, output_dir):
    fake_im_noise = torch.randn((num_samples, latent_dim), device=device)
    fake_ims = generator(fake_im_noise)
    ims = torch.clamp(fake_ims, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=nrows)
    if video_writer is None:
        tensor_shape = grid.shape
        video_writer = init_video_writer((tensor_shape[1], tensor_shape[2]))
    img = torchvision.transforms.ToPILImage()(grid)
    img = np.array(img)  # Convert PIL Image to numpy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    video_writer.write(img)
    img_output_path = os.path.join(output_dir, f'generated_sample_{generated_sample_count}.png')
    torchvision.utils.save_image(grid, img_output_path)


def main():
    parser = argparse.ArgumentParser(description="Train a GAN on the MNIST dataset")
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension size')
    parser.add_argument('--img_size', type=int, nargs=2, default=(28, 28), help='Image size (height, width)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels')
    parser.add_argument('--im_path', type=str, default='data/train/images', help='Path to image data')
    parser.add_argument('--im_ext', type=str, default='png', help='Image file extension')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=225, help='Number of samples to generate')
    parser.add_argument('--nrows', type=int, default=15, help='Number of rows for grid of generated images')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train(
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        channels=args.channels,
        im_path=args.im_path,
        im_ext=args.im_ext,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_samples=args.num_samples,
        nrows=args.nrows,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
