import os
import sys
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import DEVICE, T
from dataset import train_dataset
from unet import UNet
from diffusion import forward_diffusion
from hiq import deterministic, print_model
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a UNet model for diffusion.")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to save the trained model.')
    return parser.parse_args()



def load_model(model_path):
    """
    Load the model from the specified path. If it exists, load the state dictionary and optimizer state to resume training.
    """
    model = UNet(img_channel=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    if os.path.exists(model_path):
        print("Resuming training from existing model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        print("Starting training from scratch...")
        epoch = 0
    print_model(model)
    return model, optimizer, epoch


def train(model, optimizer, dataloader, epochs, start_epoch, lr, model_path):
    """
    Train the model with the given parameters.
    """
    loss_fn = nn.L1Loss()  # Loss function (Mean Absolute Error)
    writer = SummaryWriter()  # TensorBoard writer for logging
    n_iter = start_epoch * len(dataloader)

    model.train()
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", mininterval=30.0, file=sys.stdout):  # Progress bar for epochs
        last_loss = 0
        for batch_x, batch_cls in tqdm(dataloader, desc="Batches", leave=False, mininterval=5.0, file=sys.stdout):  # Progress bar for batches
            batch_x = batch_x.to(DEVICE) * 2 - 1  # Scale pixel values to [-1, 1]
            batch_cls = batch_cls.to(DEVICE)  # Convert class IDs to DEVICE
            batch_t = torch.randint(0, T, (batch_x.size(0),)).to(DEVICE)  # Generate random t for each image
            batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)  # Generate noisy image at time t and corresponding noise
            batch_predict_t = model(batch_x_t, batch_t, batch_cls)  # Model predicts the noise at time t
            loss = loss_fn(batch_predict_t, batch_noise_t)  # Calculate loss
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters
            last_loss = loss.item()
            writer.add_scalar('Loss/train', last_loss, n_iter)  # Log the loss
            n_iter += 1

        print(f'\nepoch:{epoch} loss={last_loss}')  # Print epoch and loss
        save_model(model, optimizer, epoch, model_path)  # Save the model


def save_model(model, optimizer, epoch, model_path):
    """
    Save the model state, optimizer state, and the current epoch to the specified path.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)


def main():
    """
    Main function to parse arguments, load data, and start the training process.
    """
    args = parse_args()

    # Data loader
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, persistent_workers=True, shuffle=True)
    model, optimizer, start_epoch = load_model(args.model_path)
    train(model, optimizer, dataloader, args.epochs, start_epoch, args.lr, args.model_path)


if __name__ == '__main__':
    main()
