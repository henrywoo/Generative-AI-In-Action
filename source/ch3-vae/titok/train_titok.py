import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision.models import inception_v3
from models import TiTok
from proxy import load_vqgan_model
from hiq.vis import print_model
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_dataset, check_pt, get_inception_score, here, load_checkpoint, save_checkpoint
import matplotlib.pyplot as plt
import torch.nn.functional as F
import signal
import sys


def commitment_loss(encoder_outputs, quantized_vectors, beta=0.25):
    return beta * F.mse_loss(quantized_vectors.detach(), encoder_outputs)


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def normalize_image(image_tensor):
    """
    Normalize the image tensor to [0, 1] range.
    """
    if image_tensor.dtype == torch.float32 or image_tensor.dtype == torch.float64:
        image_tensor = torch.clamp(image_tensor, 0, 1)
    elif image_tensor.dtype == torch.uint8:
        image_tensor = torch.clamp(image_tensor, 0, 255)
    return image_tensor


def plot_combined(image_tensor, recon_tensor, csv_file, i, task="recon"):
    plt.style.use('ggplot')
    df = pd.read_csv(csv_file)
    fig, axs = plt.subplots(2, 2, figsize=(18, 8))

    # Average Total Loss Plot
    axs[0, 0].plot(df['epoch'], df['avg_total_loss'], label='Average Total Loss', marker='o', alpha=0.5)
    axs[0, 0].plot(df['epoch'], df['recon_loss'], label='Reconstruction Loss', marker='v', alpha=0.5)
    axs[0, 0].set_title('Average Total & Reconstruction Loss over Epochs', fontsize=10)
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Average Total Loss')
    axs[0, 0].legend()

    # Reconstruction and Commitment Loss Plot
    axs[0, 1].plot(df['epoch'], df['commit_loss'], label='Commitment Loss', marker='x')
    axs[0, 1].set_title('Commitment Loss over Epochs')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()

    # Normalize and Clamp Image Tensors
    image = normalize_image(image_tensor).permute(1, 2, 0).cpu().detach().numpy()
    recon_image = normalize_image(recon_tensor).permute(1, 2, 0).cpu().detach().numpy()

    # Original Image Plot
    axs[1, 0].imshow(image)
    axs[1, 0].set_title(f"Original Image at Epoch {i}")
    axs[1, 0].axis('off')

    # Reconstructed Image Plot
    axs[1, 1].imshow(recon_image)
    axs[1, 1].set_title(f"Reconstructed Image at Epoch {i}")
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"{here}/{task}_{i}.png")
    plt.show()


def warmup_training(
        model, dataloader, optimizer, scheduler, args, device, test_loader, rank, start_epoch=0, eval_size=10
):
    model.train()
    mse_loss = nn.MSELoss()
    csv_name = f'{here}/mbin/metrics_{args.depth}_{args.heads}_{args.mlp_dim}_{args.image_size}_{args.K}.csv'
    metrics_file = os.path.join(args.log_dir, csv_name)

    check_fid = False
    if check_fid and rank == 0:
        inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
        real_features = []
        count = 0
        for images, _ in tqdm(test_loader):
            if count >= eval_size:
                break
            images = images.to(device)
            with torch.no_grad():
                features = inception_model(images)
            real_features.append(features)
            count += images.size(0)
        real_features = torch.cat(real_features, dim=0).cpu().numpy()

    beta = 0.25
    for epoch in range(start_epoch, args.num_epochs_warmup):
        total_loss = 0
        total_recon_loss = 0
        total_commit_loss = 0
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, quantized_tokens, encoder_outputs = model(images)
            recon_loss = mse_loss(reconstructed, images)
            commit_loss = commitment_loss(encoder_outputs, quantized_tokens, beta)
            loss = recon_loss + commit_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
        scheduler.step()
        avg_total_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_commit_loss = total_commit_loss / len(dataloader)
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(
                f'Epoch [{epoch + 1}/{args.num_epochs_warmup}], Total Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, Commit Loss: {avg_commit_loss:.4f}, LR: {lr:.6f}')

            # Save checkpoint
            checkpoint_path = args.resume
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                filename=checkpoint_path,
            )
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
            else:
                metrics_df = pd.DataFrame(
                    columns=['epoch', 'avg_total_loss', 'recon_loss', 'commit_loss', 'learning_rate'])

            new_row_df = pd.DataFrame(
                {'epoch': [epoch + 1], 'avg_total_loss': [avg_total_loss], 'recon_loss': [avg_recon_loss],
                 'commit_loss': [avg_commit_loss], 'learning_rate': [lr]})
            metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
            metrics_df.to_csv(metrics_file, index=False)

            if (epoch + 1) % 1 == 0:
                #print("labels[0]:", labels[0].item())
                plot_combined(images[0], reconstructed[0], metrics_file, epoch + 1)

            if check_fid:
                model.eval()
                fake_images = []
                count = 0
                with torch.no_grad():
                    for images, _ in test_loader:
                        if count >= eval_size:
                            break
                        images = images.to(device)
                        reconstructed, _ = model(images)
                        fake_images.append(reconstructed)
                        count += images.size(0)
                fake_images = torch.cat(fake_images, dim=0)
                fake_features = []
                for i in range(0, len(fake_images), args.batch_size):
                    batch = fake_images[i: i + args.batch_size].to(device)
                    with torch.no_grad():
                        features = inception_model(batch)
                    fake_features.append(features)
                fake_features = torch.cat(fake_features, dim=0).cpu().numpy()
                fid = calculate_fid(real_features, fake_features)
                is_mean, is_std = get_inception_score(fake_images, inception_model)
                print(f"FID: {fid}, IS: {is_mean} ± {is_std}")


def main(rank, args):
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C! Shutting down gracefully...')
        dist.destroy_process_group()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    train_loader, test_loader = get_dataset(
        args.dataset, args.image_size, args.batch_size, args.data_dir, rank, args.world_size
    )
    vqgan_model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

    if args.image_size == 256:
        args.patch_size = 16
    elif args.image_size == 512:
        args.patch_size = 32
    else:
        raise ValueError("Unsupported image size. Supported sizes are 256 and 512.")

    model = TiTok(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        K=args.K,
        codebook=vqgan_model.quantize.embedding.weight,
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print_model(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs_warmup)

    # Load checkpoint if exists
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        model, optimizer, start_epoch = load_checkpoint(args.resume, model, optimizer)

    warmup_training(model, train_loader, optimizer, scheduler, args, device, test_loader, rank, start_epoch=start_epoch)

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train TiTok Model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs_warmup', type=int, default=101, help='Number of warmup epochs')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimensionality of the latent space')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of each image patch')
    parser.add_argument('--depth', type=int, default=12, help='Depth of the transformer')
    parser.add_argument('--heads', type=int, default=6, help='Number of heads in multi-head attention')
    parser.add_argument('--mlp_dim', type=int, default=2048, help='Dimensionality of the MLP in the transformer')
    parser.add_argument('--K', type=int, default=32, help='Number of latent tokens')
    parser.add_argument(
        '--vqgan_config',
        type=str,
        default=f'{here}/pretrained_maskgit/VQGAN/model.yaml',
        help='Path to VQGAN config file',
    )
    parser.add_argument(
        '--vqgan_checkpoint',
        type=str,
        default=f'{here}/pretrained_maskgit/VQGAN/last.ckpt',
        help='Path to VQGAN checkpoint file',
    )
    parser.add_argument('--log_dir', type=str, default=f'{here}/logs', help='Directory for logs')
    parser.add_argument('--data_dir', type=str, default="data", help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    parser.add_argument(
        '--resume', type=str, default='', help='Path to the latest checkpoint'
    )

    args = parser.parse_args()
    if args.resume == '':
        args.resume = f'{here}/mbin/checkpoint_{args.depth}_{args.heads}_{args.mlp_dim}_{args.image_size}_{args.K}.pth.tar'
    check_pt()

    if torch.cuda.device_count() > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)
