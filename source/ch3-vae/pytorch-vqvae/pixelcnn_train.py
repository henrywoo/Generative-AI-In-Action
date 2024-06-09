import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from modules import VectorQuantizedVAE, VQEmbedding, ResBlock, weights_init  # Ensure this imports your VQ-VAE model definition
import matplotlib.pyplot as plt
from tqdm import tqdm

file_vqvae = 'models/vqvae/best.pt'
file_codebook = 'models/vqvae/latent_codes.pt'
file_pixelcnn = 'models/vqvae/best_pixelcnn.pt'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import argparse
import os
from PIL import Image

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class AutoregressivePixelCNN(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(AutoregressivePixelCNN, self).__init__()
        self.conv1 = MaskedConv2d('A', embed_dim, 128, kernel_size=7, padding=3)
        self.conv2 = MaskedConv2d('B', 128, 128, kernel_size=7, padding=3)
        self.conv3 = MaskedConv2d('B', 128, 128, kernel_size=7, padding=3)
        self.conv4 = MaskedConv2d('B', 128, num_embeddings, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

def extract_latent_codes(vqvae_model, dataloader, device):
    vqvae_model.eval()
    all_latents = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            z_e_x = vqvae_model.encoder(data)
            z_q_x = vqvae_model.codebook(z_e_x).argmax(dim=1)
            all_latents.append(z_q_x.cpu())

    all_latents = torch.cat(all_latents)
    return all_latents

def prepare_dataset(vqvae_model, dataloader, device):
    latent_codes = extract_latent_codes(vqvae_model, dataloader, device)
    latent_codes = latent_codes.unsqueeze(1)  # Ensure shape is (N, 1, 7, 7)
    print(f"Latent codes shape: {latent_codes.shape}")  # Debugging print
    dataset = TensorDataset(latent_codes)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_pixelcnn(pixelcnn_model, dataloader, device, epochs=10, lr=1e-3):
    optimizer = optim.Adam(pixelcnn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    pixelcnn_model.to(device)

    all_losses = []

    for epoch in range(epochs):
        pixelcnn_model.train()
        total_loss = 0

        for batch in dataloader:
            latents = batch[0].to(device)  # (batch_size, 1, 7, 7)
            latents = latents.long()  # Ensure latents are long type for cross-entropy loss
            print(f"Latents shape: {latents.shape}")  # Debugging print

            optimizer.zero_grad()
            output = pixelcnn_model(latents.float())  # Convert latents to float for convolutional layers
            print(f"Output shape: {output.shape}")  # Debugging print

            # Flatten the output and target for cross-entropy loss
            output = output.permute(0, 2, 3, 1).contiguous()  # Shape: (batch_size, 7, 7, num_embeddings)
            output = output.view(-1, output.size(-1))  # Shape: (batch_size * 7 * 7, num_embeddings)
            latents = latents.view(-1)  # Shape: (batch_size * 7 * 7)

            loss = criterion(output, latents)  # Compute cross-entropy loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        average_loss = total_loss / len(dataloader)
        all_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

    return all_losses

def sample_from_pixelcnn(model, shape, device):
    model.eval()
    samples = torch.zeros(shape, device=device, dtype=torch.long)  # Initialize as long for indices
    with torch.no_grad():
        for i in range(shape[2]):  # Height dimension
            for j in range(shape[3]):  # Width dimension
                out = model(samples.float())  # Convert samples to float for model input
                probs = F.softmax(out[:, :, i, j], dim=1)  # Get probabilities
                samples[:, 0, i, j] = torch.multinomial(probs, 1).squeeze(-1)
    return samples

def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_samples_from_pixelcnn(pixelcnn_model, vqvae_model, args):
    with torch.no_grad():
        # Sample from PixelCNN to get indices
        z_q_indices = sample_from_pixelcnn(pixelcnn_model, (args.batch_size, 1, 7, 7), args.device)
        print(f"z_q_indices shape: {z_q_indices.shape}")  # Debugging print
        z_q_indices = z_q_indices.squeeze(1)  # Shape: (batch_size, 7, 7)
        print(f"z_q_indices shape after squeeze: {z_q_indices.shape}")  # Debugging print

        # Convert indices to embeddings using the codebook
        z_q = vqvae_model.codebook.embedding(z_q_indices).permute(0, 3, 1, 2).contiguous()
        print(f"z_q shape after embedding and permute: {z_q.shape}")  # Debugging print

        # Decode the latent vectors to images
        x_tilde = vqvae_model.decode(z_q)

        # Upscale images to 256x256
        upscaled_images = []
        for img in x_tilde:
            img_pil = transforms.ToPILImage()(img.cpu())
            img_upscaled = img_pil.resize((256, 256), Image.BILINEAR)  # Resize to 256x256
            upscaled_images.append(transforms.ToTensor()(img_upscaled))
        upscaled_images = torch.stack(upscaled_images)
    return upscaled_images

def main(args):
    if args.dataset in ['mnist', 'fashion-mnist']:
        num_channels = 1
    else:
        num_channels = 3
    vqvae_model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k)
    vqvae_model = load_model(vqvae_model, args.vqvae_model_path, args.device)

    # Prepare dataset for PixelCNN training
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    latent_dataloader = prepare_dataset(vqvae_model, dataloader, args.device)

    pixelcnn_model = AutoregressivePixelCNN(args.k, 1)  # Use the new PixelCNN class with 1 channel input
    all_losses = train_pixelcnn(pixelcnn_model, latent_dataloader, args.device, epochs=30, lr=1e-3)

    # Save the trained PixelCNN model
    torch.save(pixelcnn_model.state_dict(), args.pixelcnn_model_path)

    # Generate samples from the trained PixelCNN
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    num_batches = args.num_images // args.batch_size
    if args.num_images % args.batch_size != 0:
        num_batches += 1

    for i in range(num_batches):  # Generate specified number of images
        x_tilde = generate_samples_from_pixelcnn(pixelcnn_model, vqvae_model, args)
        grid = make_grid(x_tilde.cpu(), nrow=8, range=(-1, 1), normalize=True)
        save_image(grid, os.path.join(args.output_folder, f'generated_samples_{i}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQ-VAE + PixelCNN Inference')
    parser.add_argument('--dataset', default='fashion-mnist', help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--hidden-size', type=int, default=256, help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512, help='number of latent vectors (default: 512)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument('--num-images', type=int, default=64, help='number of images to generate (default: 64)')
    parser.add_argument('--output-folder', type=str, default='vqvae_infer', help='name of the output folder (default: vqvae_infer)')
    parser.add_argument('--vqvae_model_path', type=str, default="models/vqvae/best.pt", help='path to the trained VQ-VAE model file (e.g., best_vqvae.pt)')
    parser.add_argument('--pixelcnn_model_path', type=str, default="models/vqvae/best_pixelcnn.pt", help='path to the trained PixelCNN model file (e.g., best_pixelcnn.pt)')
    parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cuda)')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    main(args)
