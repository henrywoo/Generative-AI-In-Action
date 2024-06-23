import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()
        self.best_loss = float('inf')

        # Load existing VQGAN checkpoint
        #if os.path.exists(args.checkpoint_path):
        #    self.load_vqgan_checkpoint(args.checkpoint_path)

        # Load transformer checkpoint if it exists
        if os.path.exists(args.tr_checkpoint_path):
            self.load_checkpoint(args.tr_checkpoint_path)

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def load_vqgan_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        #self.model.load_state_dict(checkpoint['model_state_dict'])
        print("VQGAN checkpoint loaded.")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Checkpoint loaded. Resuming training from epoch {checkpoint['epoch']}.")

    def save_checkpoint(self, epoch, best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'best_loss': self.best_loss
        }
        filename = 'transformer_best.pt' if best else f"transformer_{epoch}.pt"
        torch.save(checkpoint, os.path.join("checkpoints", filename))

    def validate(self, val_dataset):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs in val_dataset:
                imgs = imgs[0].to(device=args.device)
                logits, targets = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_dataset)
        self.model.train()
        return val_loss

    def train(self, args):
        train_dataset = load_data(args)
        val_dataset = load_data(args, train=False)  # Assuming load_data can handle train/validation split
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs = imgs[0].to(device=args.device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)

            val_loss = self.validate(val_dataset)
            print(f"Epoch {epoch}, Validation Loss: {val_loss}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, best=True)

            if epoch % 20 == 0:
                log, sampled_imgs = self.model.log_images(imgs[0][None])
                vutils.save_image(sampled_imgs, os.path.join("results", f"transformer_{epoch}.jpg"), nrow=4)
                plot_images(log)

            #self.save_checkpoint(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/vqgan_best.pt',
                        help='Path to VQGAN checkpoint.')
    parser.add_argument('--tr_checkpoint_path', type=str, default='./checkpoints/last_ckpt.pt',
                        help='Path to transformer checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=20, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1.,
                        help='Weighting factor for perceptual loss.')
    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.checkpoint_path = "./checkpoints/vqgan_best.pt"
    args.tr_checkpoint_path = "./checkpoints/last_ckpt.pt"

    train_transformer = TrainTransformer(args)
