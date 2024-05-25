import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import logging
import os
from dataset import MNIST  # Assuming this is a custom module
from vit import ViT  # Assuming this is a custom module
from torch.optim.lr_scheduler import CosineAnnealingLR
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on MNIST")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, default="model_best.pth", help="Path to save the model")
    parser.add_argument("--val_split", type=float, default=0.05, help="Proportion of data to use for validation")
    return parser.parse_args()

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {DEVICE}")
    full_dataset = MNIST()
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, persistent_workers=True)
    model = ViT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    try:
        model.load_state_dict(torch.load(args.model_path))
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.warning("No saved model found, starting from scratch.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    ema_model = deepcopy(model)
    alpha = 0.99  # decay factor for EMA
    best_val_loss = float('inf')
    training_losses, validation_losses = train_and_validate_model(model, ema_model, optimizer, scheduler, alpha, train_loader, val_loader, DEVICE, args, best_val_loss)
    plot_losses(training_losses, validation_losses, args.model_path.replace('.pth', '_loss.png'))

def update_ema(ema_model, model, alpha):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

def train_and_validate_model(model, ema_model, optimizer, scheduler, alpha, train_loader, val_loader, device, args, best_val_loss):
    training_losses = []
    validation_losses = []
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        count = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model, alpha)
            total_train_loss += loss.item()
            count += 1
        avg_train_loss = total_train_loss / count
        training_losses.append(avg_train_loss)
        model.eval()
        total_val_loss = 0
        count = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = ema_model(imgs)
                loss = F.cross_entropy(logits, labels)
                total_val_loss += loss.item()
                count += 1
        avg_val_loss = total_val_loss / count
        validation_losses.append(avg_val_loss)
        logging.info(f'Epoch: {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ema_model.state_dict(), args.model_path)
            logging.info("Saved new best model.")
    return training_losses, validation_losses

def plot_losses(training_losses, validation_losses, output_path):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, marker='o', label='Training Loss per Epoch', alpha=0.5)
    plt.plot(validation_losses, marker='x', label='Validation Loss per Epoch', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)
