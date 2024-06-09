import torch
from torchvision.utils import save_image, make_grid
from modules import VectorQuantizedVAE
import argparse
import os
from pixelcnn import PixelCNN

def sample_from_pixelcnn(model, shape, device):
    model.eval()
    samples = torch.zeros(shape, device=device, dtype=torch.long)
    with torch.no_grad():
        for i in range(shape[2]):
            for j in range(shape[3]):
                out = model(samples)
                probs = F.softmax(out[:, :, i, j], dim=1)
                samples[:, :, i, j] = torch.multinomial(probs.view(shape[0], -1), 1).view(-1)
    return samples


def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    return model

from torchvision import transforms
from PIL import Image

def generate_samples_from_pixelcnn(pixelcnn_model, vqvae_model, args):
    with torch.no_grad():
        z_q_indices = sample_from_pixelcnn(pixelcnn_model, (args.batch_size, args.hidden_size, 8, 8), args.device)
        z_q = vqvae_model.codebook.embedding(z_q_indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_tilde = vqvae_model.decode(z_q)

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

    pixelcnn_model = PixelCNN(args.k, args.hidden_size)
    pixelcnn_model = load_model(pixelcnn_model, args.pixelcnn_model_path, args.device)

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
    parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)', required=True)
    parser.add_argument('--hidden-size', type=int, default=256, help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=512, help='number of latent vectors (default: 512)')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument('--num-images', type=int, default=64, help='number of images to generate (default: 64)')
    parser.add_argument('--output-folder', type=str, default='vqvae_infer', help='name of the output folder (default: vqvae_infer)')
    parser.add_argument('--vqvae_model_path', type=str, required=True, help='path to the trained VQ-VAE model file (e.g., best_vqvae.pt)')
    parser.add_argument('--pixelcnn_model_path', type=str, required=True, help='path to the trained PixelCNN model file (e.g., best_pixelcnn.pt)')
    parser.add_argument('--device', type=str, default='cuda', help='set the device (cpu or cuda, default: cuda)')

    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    main(args)
