import torch
import os
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
from models import Generator, Discriminator
import matplotlib.pyplot as plt  # Import Matplotlib
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_video_writer(frame_size, output_file='gan_training.mp4', fps=2.0):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    return video_writer


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

    # Initialize lists to store losses and metrics
    generator_losses_per_epoch = []
    discriminator_losses_per_epoch = []
    total_minimax_losses_per_epoch = []
    fid_scores_per_epoch = []
    inception_scores_per_epoch = []

    # Initialize FID and IS metrics
    fid = FrechetInceptionDistance(feature=64).to(device)
    inception_score = InceptionScore(feature=64).to(device)

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
                    infer_for_video(generated_sample_count, generator, latent_dim, num_samples, nrows, video_writer,
                                    output_dir)
                    generated_sample_count += 1
                    generator.train()
            steps += 1

            # Update FID with real images
            real_ims_for_fid = (real_ims + 1) / 2  # Scale to [0, 1]
            real_ims_for_fid = (real_ims_for_fid * 255).byte()  # Scale to [0, 255] and convert to uint8
            real_ims_for_fid = real_ims_for_fid.repeat(1, 3, 1, 1)  # Convert to 3 channels
            fid.update(real_ims_for_fid, real=True)

        # Compute mean losses for the epoch
        mean_gen_loss = np.mean(generator_losses)
        mean_disc_loss = np.mean(discriminator_losses)

        # Compute total minimax loss for the epoch
        total_minimax_loss = mean_gen_loss + mean_disc_loss
        total_minimax_losses_per_epoch.append(total_minimax_loss)

        # Append mean losses for the epoch
        generator_losses_per_epoch.append(mean_gen_loss)
        discriminator_losses_per_epoch.append(mean_disc_loss)

        # Calculate FID and IS scores
        fake_ims_for_metrics = generate_fake_samples(generator, latent_dim, num_samples)
        fake_ims_for_metrics = (fake_ims_for_metrics * 255).byte()  # Scale to [0, 255] and convert to uint8
        fake_ims_for_metrics = fake_ims_for_metrics.repeat(1, 3, 1, 1)  # Convert to 3 channels

        fid.update(fake_ims_for_metrics, real=False)
        inception_score.update(fake_ims_for_metrics)

        fid_score = fid.compute().item()
        inception_score_val = inception_score.compute()[0].item()  # Only take the first value

        fid_scores_per_epoch.append(fid_score)
        inception_scores_per_epoch.append(inception_score_val)

        # Reset the metrics for the next epoch
        fid.reset()
        inception_score.reset()

        torch.save(generator.state_dict(), os.path.join(output_dir, 'generator_ckpt.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_ckpt.pth'))

    print('Done Training ...')
    if video_writer:
        video_writer.release()

    # Plot the losses and metrics
    plot_separate_losses(generator_losses_per_epoch, discriminator_losses_per_epoch, output_dir)
    plot_combined_losses(generator_losses_per_epoch, discriminator_losses_per_epoch, total_minimax_losses_per_epoch,
                         output_dir)
    plot_metrics(fid_scores_per_epoch, inception_scores_per_epoch, output_dir)


def generate_fake_samples(generator, latent_dim, num_samples):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim, device=device)
        fake_samples = generator(noise)
        fake_samples = (fake_samples + 1) / 2  # Scale to [0, 1]
        fake_samples = (fake_samples * 255).byte()  # Scale to [0, 255] and convert to uint8
    return fake_samples


def infer_for_video(generated_sample_count, generator, latent_dim, num_samples, nrows, video_writer, output_dir,
                    save_img=False):
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
    if save_img:
        img_output_path = os.path.join(output_dir, f'generated_sample_{generated_sample_count}.png')
        torchvision.utils.save_image(grid, img_output_path)


def plot_separate_losses(generator_losses, discriminator_losses, output_dir):
    epochs = range(len(generator_losses))
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, generator_losses, label='Generator Loss', alpha=0.75, linestyle='-.')
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Loss', fontsize=8)
    plt.title('Generator Loss vs Epochs', fontsize=10)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, discriminator_losses, label='Discriminator Loss', alpha=0.75, linestyle='--')
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Loss', fontsize=8)
    plt.title('Discriminator Loss vs Epochs', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generator_discriminator_losses_vs_epochs.png'))
    plt.show()


def plot_combined_losses(generator_losses, discriminator_losses, total_minimax_losses, output_dir):
    epochs = range(len(generator_losses))
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 3.6))
    plt.plot(epochs, generator_losses, label='Generator Loss', alpha=0.75, linestyle='-.')
    plt.plot(epochs, discriminator_losses, label='Discriminator Loss', alpha=0.75, linestyle='--')
    plt.plot(epochs, total_minimax_losses, label='Total Minimax Loss')
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Loss', fontsize=8)
    plt.title('Losses vs Epochs', fontsize=10)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_losses_vs_epochs.png'))
    plt.show()


def plot_metrics(fid_scores, inception_scores, output_dir):
    epochs = range(len(fid_scores))
    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4.8))  # Adjusted figure size for better visualization

    # FID Score subplot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, fid_scores, label='FID Score', alpha=0.75, linestyle='-.')
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('FID Score', fontsize=8)
    plt.title('FID Score vs Epochs', fontsize=10)
    plt.legend()

    # Inception Score subplot
    plt.subplot(2, 1, 2)
    plt.plot(epochs, inception_scores, label='Inception Score', alpha=0.75, linestyle='--')
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Inception Score', fontsize=8)
    plt.title('Inception Score vs Epochs', fontsize=10)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fid_inception_scores_vs_epochs.png'))
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Train a GAN on the MNIST dataset")
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension size')
    parser.add_argument('--img_size', type=int, nargs=2, default=(28, 28), help='Image size (height, width)')
    parser.add_argument('--channels', type=int, default=1, help='Number of image channels')
    parser.add_argument('--im_path', type=str, default='data/train/images', help='Path to image data')
    parser.add_argument('--im_ext', type=str, default='png', help='Image file extension')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
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
    set_seed(has_torch=True)
    main()
