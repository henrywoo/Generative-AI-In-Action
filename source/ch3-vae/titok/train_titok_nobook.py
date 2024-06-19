import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from proxy import load_vqgan_model
from hiq import ensure_folder
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_dataset, check_pt, here, save_checkpoint
import matplotlib.pyplot as plt
import signal
import sys

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losses = checkpoint.get('losses', [])
        print(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return model, optimizer, start_epoch, losses
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0, []

def plot_attention(image, attn):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize to [0, 1]
    pad = ((0, 32), (0, 32), (0, 0))  # Pad bottom, right, and no padding for channels
    expanded_image_np = np.pad(image_np, pad_width=pad, mode='constant', constant_values=0)
    attn_resized = attn.cpu().detach().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(expanded_image_np)
    plt.imshow(attn_resized, cmap='Reds', alpha=0.3)  # alpha controls the transparency of the overlay
    plt.colorbar(shrink=0.3, aspect=10)
    plt.title('Attention Map Overlay', fontsize=8)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def plot_losses(losses, output_path):
    plt.style.use('ggplot')
    plt.plot(losses, marker='o', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(output_path)
    plt.show()

def plot_reconstructions(model, test_loader, device):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        reconstructions, _ = model(images)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(4):
        axes[0, i].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')
    plt.tight_layout()
    plt.savefig('reconstructions_nobook.png')
    plt.show()

def warmup_training(
        model, dataloader, optimizer, scheduler, args, device, test_loader, rank, vqgan_model, start_epoch=0, eval_size=10, losses=[]
):
    model.train()
    mse_loss = nn.MSELoss()
    csv_name = f'{here}/mbin-nobook/{args.model_type}/metrics_{args.depth}_{args.heads}_{args.mlp_dim}_{args.image_size}_{args.K}.csv'
    ensure_folder(csv_name)
    metrics_file = os.path.join(args.log_dir, csv_name)

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        showed_attention = False
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            if i == len(dataloader) - 1:
                break
            images = images.to(device)
            optimizer.zero_grad()
            d, _ = model(images)
            if os.getenv('DEBUG', 'False').lower() in ('true', '1', 't') and not showed_attention:
                attn_weights = model.module.encoder.transformer.layers[-1][0].attn_weights
                print(attn_weights.shape) # torch.Size([128, 3, 288, 288])
                plot_attention(images[0], attn_weights[0, 0])
                showed_attention = True
            loss = mse_loss(d, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_total_loss = total_loss / len(dataloader)
        losses.append(avg_total_loss)
        lr = scheduler.get_last_lr()[0]
        if rank == 0:
            print(f'Epoch [{epoch + 1}/{args.epochs}], Total Loss: {avg_total_loss:.4f}, LR: {lr:.6f}')
            # Save checkpoint
            checkpoint_path = args.resume
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'losses': losses
                },
                filename=checkpoint_path,
            )
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
            else:
                metrics_df = pd.DataFrame(
                    columns=['epoch', 'avg_total_loss', 'learning_rate'])

            new_row_df = pd.DataFrame({'epoch': [epoch + 1], 'avg_total_loss': [avg_total_loss], 'learning_rate': [lr]})
            metrics_df = pd.concat([metrics_df, new_row_df], ignore_index=True)
            metrics_df.to_csv(metrics_file, index=False)

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

    from models_no_codebook import TiTok

    model = TiTok(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        K=args.K,
        B=args.batch_size,
        codebook=vqgan_model.quantize.embedding.weight,
    ).to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Load checkpoint if exists
    start_epoch = 0
    losses = []
    if args.resume and os.path.isfile(args.resume):
        model, optimizer, start_epoch, losses = load_checkpoint(args.resume, model, optimizer)

    warmup_training(model, train_loader, optimizer, scheduler, args, device, test_loader, rank, vqgan_model=vqgan_model, start_epoch=start_epoch, losses=losses)

    if rank == 0:
        plot_losses(losses, os.path.join(args.log_dir, 'training_losses.png'))
        plot_reconstructions(model, test_loader, device)

    dist.destroy_process_group()

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Train TiTok Model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimensionality of the latent space')
    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--patch_size', type=int, default=32, help='Size of each image patch')
    parser.add_argument('--depth', type=int, default=12, help='Depth of the transformer')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads in multi-head attention')
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
    parser.add_argument('--model_type', type=str, default="base", help='first version')
    parser.add_argument(
        '--resume', type=str, default='', help='Path to the latest checkpoint'
    )

    args = parser.parse_args()
    if args.resume == '':
        args.resume = f'{here}/mbin-nobook/{args.model_type}/checkpoint_{args.depth}_{args.heads}_{args.mlp_dim}_{args.image_size}_{args.K}.pth.tar'
        ensure_folder(args.resume)
    check_pt()

    if torch.cuda.device_count() > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)
