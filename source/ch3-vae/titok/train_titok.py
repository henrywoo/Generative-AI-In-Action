import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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


def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def warmup_training(model, dataloader, optimizer, scheduler, args, device, writer, test_loader, eval_size=10):
    model.train()
    mse_loss = nn.MSELoss()
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)
    # Collect real features from the test set using only eval_size images
    real_features = []
    count = 0
    for images, _ in tqdm(test_loader):
        if count >= eval_size:
            break
        images = images.to(device)
        with torch.no_grad():
            features = inception_model(images).cpu().numpy()
        real_features.append(features)
        count += images.size(0)
    real_features = np.concatenate(real_features, axis=0)

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
        print(f'Epoch [{epoch+1}/{args.num_epochs_warmup}], Loss: {avg_loss:.4f}')
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        if epoch % 5 == 0:
            model.eval()
            fake_images = []
            count = 0
            with torch.no_grad():
                for images, _ in test_loader:
                    if count >= eval_size:
                        break
                    images = images.to(device)
                    reconstructed, _ = model(images)
                    fake_images.append(reconstructed.cpu())
                    count += images.size(0)
            fake_images = torch.cat(fake_images, 0).to(device)
            fake_features = []
            for i in range(0, len(fake_images), args.batch_size):
                batch = fake_images[i : i + args.batch_size].to(device)
                with torch.no_grad():
                    features = inception_model(batch).cpu().numpy()
                fake_features.append(features)
            fake_features = np.concatenate(fake_features, axis=0)
            fid = calculate_fid(real_features, fake_features)
            is_mean, is_std = get_inception_score(fake_images, inception_model)
            print(f"FID: {fid}, IS: {is_mean} ± {is_std}")
            writer.add_scalar("FID", fid, epoch)
            writer.add_scalar("IS", is_mean, epoch)
            model.train()


def main(args):
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # Load dataset
    train_loader, test_loader = get_dataset(args.dataset, args.image_size, args.batch_size, args.data_dir, local_rank, args.world_size)

    # Load VQGAN model
    vqgan_model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)

    # Adjust patch size based on image size
    if args.image_size == 256:
        patch_size = 16
    elif args.image_size == 512:
        patch_size = 32
    else:
        raise ValueError("Unsupported image size. Supported sizes are 256 and 512.")

    # Initialize TiTok model
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

    # Wrap the model with DDP for multi-GPU training
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    print_model(model)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs_warmup)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir) if local_rank == 0 else None

    # Start warmup training
    warmup_training(model, train_loader, optimizer, scheduler, args, device, writer, test_loader)

    if writer:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TiTok Model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_epochs_warmup', type=int, default=10, help='Number of warmup epochs')
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
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for TensorBoard logs')
    parser.add_argument('--data_dir', type=str, default="data", help='Directory containing the dataset')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--world_size', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')

    args = parser.parse_args()
    check_pt()
    # Launch distributed processes
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)
