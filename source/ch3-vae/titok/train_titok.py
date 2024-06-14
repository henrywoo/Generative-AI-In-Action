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
from utils import get_dataset, check_pt, get_inception_score, here
import matplotlib.pyplot as plt


def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def warmup_training(model, dataloader, optimizer, scheduler, args, device, test_loader, rank, eval_size=10):
    model.train()
    mse_loss = nn.MSELoss()
    metrics_file = os.path.join(args.log_dir, 'metrics.csv')

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

    for epoch in range(args.num_epochs_warmup):
        total_loss = 0
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, quantized_tokens = model(images)
            loss = mse_loss(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs_warmup}], Loss: {avg_loss:.4f}, LR: {lr:.6f}')

            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
            else:
                metrics_df = pd.DataFrame(columns=['epoch', 'avg_loss', 'learning_rate'])

            metrics_df = metrics_df.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'learning_rate': lr}, ignore_index=True)
            metrics_df.to_csv(metrics_file, index=False)

            if (epoch + 1) % 5 == 0:
                plot_metrics(metrics_file)

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
                    batch = fake_images[i : i + args.batch_size].to(device)
                    with torch.no_grad():
                        features = inception_model(batch)
                    fake_features.append(features)
                fake_features = torch.cat(fake_features, dim=0).cpu().numpy()
                fid = calculate_fid(real_features, fake_features)
                is_mean, is_std = get_inception_score(fake_images, inception_model)
                print(f"FID: {fid}, IS: {is_mean} ± {is_std}")


def plot_metrics(csv_file):
    df = pd.read_csv(csv_file)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(df['epoch'], df['avg_loss'], label='Average Loss')
    axs[0].set_title('Average Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Average Loss')
    axs[0].legend()

    axs[1].plot(df['epoch'], df['learning_rate'], label='Learning Rate', color='orange')
    axs[1].set_title('Learning Rate over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Learning Rate')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main(rank, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    train_loader, test_loader = get_dataset(args.dataset, args.image_size, args.batch_size, args.data_dir, rank, args.world_size)
    vqgan_model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

    if args.image_size == 256:
        patch_size = 16
    elif args.image_size == 512:
        patch_size = 32
    else:
        raise ValueError("Unsupported image size. Supported sizes are 256 and 512.")

    model = TiTok(
        image_size=args.image_size,
        patch_size=patch_size,
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

    warmup_training(model, train_loader, optimizer, scheduler, args, device, test_loader, rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train TiTok Model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs_warmup', type=int, default=100, help='Number of warmup epochs')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimensionality of the latent space')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of each image patch')
    parser.add_argument('--depth', type=int, default=6, help='Depth of the transformer')
    parser.add_argument('--heads', type=int, default=16, help='Number of heads in multi-head attention')
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
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logs')
    parser.add_argument('--data_dir', type=str, default="data", help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')

    args = parser.parse_args()
    check_pt()

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)
