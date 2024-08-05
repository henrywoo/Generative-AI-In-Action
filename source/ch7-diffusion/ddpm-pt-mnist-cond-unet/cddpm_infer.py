import os
import argparse
import torch
from torch import nn
from torchvision.utils import save_image
from config import DEVICE, T, IMG_SIZE
from unet import UNet
from diffusion import *
from lora import LoraLayer, inject_lora
from hiq import print_model


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate images using a trained UNet model for diffusion.")
    parser.add_argument('--model_path', type=str, default="model.pt", help='Path to the trained model.')
    parser.add_argument('--output_dir', type=str, default='output_images', help='Directory to save generated images.')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate.')
    parser.add_argument('--class_label', type=int, default=0, help='Class label to guide image generation.')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for inference.')
    parser.add_argument('--sampling_method', type=str, default='ddim', choices=['ddpm', 'ddim'], help='Sampling method to use (ddpm or ddim).')
    return parser.parse_args()


def load_model(model_path, use_lora=False):
    """
    Load the trained model from the specified path.
    """
    model = UNet(img_channel=1).to(DEVICE)
    if os.path.exists(model_path):
        print("Loading trained model...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if use_lora:
        # Inject LoRA layers
        for name, layer in model.named_modules():
            name_cols = name.split('.')
            filter_names = ['w_q', 'w_k', 'w_v']
            if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
                inject_lora(model, name, layer)

        # Load LoRA weights
        try:
            restore_lora_state = torch.load('lora.pt')
            model.load_state_dict(restore_lora_state, strict=False)
        except:
            pass

        model = model.to(DEVICE)

        # Merge LoRA weights into the main model
        for name, layer in model.named_modules():
            name_cols = name.split('.')
            if isinstance(layer, LoraLayer):
                children = name_cols[:-1]
                cur_layer = model
                for child in children:
                    cur_layer = getattr(cur_layer, child)
                lora_weight = (layer.lora_a @ layer.lora_b) * layer.alpha / layer.r
                before_weight = layer.raw_linear.weight.clone()
                layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(lora_weight.T)).to(DEVICE)
                setattr(cur_layer, name_cols[-1], layer.raw_linear)

    print_model(model)
    return model


def backward_denoise_ddpm(model, batch_x_t, batch_cls):
    steps = [batch_x_t, ]
    global alphas, alphas_cumprod, variance

    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    variance = variance.to(DEVICE)
    batch_cls = batch_cls.to(DEVICE)

    model.eval()
    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            batch_t = torch.full((batch_x_t.size(0),), t).to(DEVICE)
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_cls)
            shape = (batch_x_t.size(0), 1, 1, 1)
            batch_mean_t = 1 / torch.sqrt(alphas[batch_t].view(*shape)) * (
                    batch_x_t -
                    (1 - alphas[batch_t].view(*shape)) / torch.sqrt(
                1 - alphas_cumprod[batch_t].view(*shape)) * batch_predict_noise_t
            )
            if t != 0:
                batch_x_t = batch_mean_t + torch.randn_like(batch_x_t) * torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t = batch_mean_t
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)
    return steps


def backward_denoise_ddim(model, batch_x_t, batch_cls, ddim_steps=50, eta=0.0):
    steps = [batch_x_t, ]
    global alphas, alphas_cumprod, betas

    model = model.to(DEVICE)
    batch_x_t = batch_x_t.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alphas_cumprod = alphas_cumprod.to(DEVICE)
    betas = betas.to(DEVICE)
    batch_cls = batch_cls.to(DEVICE)

    model.eval()
    with torch.no_grad():
        ddim_timesteps = torch.linspace(0, T - 1, steps=ddim_steps).to(torch.int64).to(DEVICE)
        for i in range(len(ddim_timesteps) - 1, -1, -1):
            t = ddim_timesteps[i]
            batch_t = torch.full((batch_x_t.size(0),), t).to(DEVICE)
            batch_predict_noise_t = model(batch_x_t, batch_t, batch_cls)
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_next = alphas_cumprod[ddim_timesteps[i - 1]] if i > 0 else torch.tensor(1.0).to(DEVICE)

            batch_mean_t = (
                torch.sqrt(alpha_cumprod_next) *
                (
                    (batch_x_t - torch.sqrt(1 - alpha_cumprod_t) * batch_predict_noise_t) /
                    torch.sqrt(alpha_cumprod_t)
                ) +
                torch.sqrt(1 - alpha_cumprod_next) * batch_predict_noise_t
            )

            if eta > 0:
                z = torch.randn_like(batch_x_t)
                sigma_t = eta * torch.sqrt((1 - alpha_cumprod_next) / (1 - alpha_cumprod_t) * (1 - alpha_t))
                batch_mean_t += sigma_t * z

            batch_x_t = batch_mean_t
            batch_x_t = torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)
    return steps


def generate_images(model_path, num_images, class_label, output_dir, use_lora, sampling_method):
    """
    Generate images using the trained model.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_path, use_lora)

    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)
            class_tensor = torch.tensor([class_label], dtype=torch.long).to(DEVICE)

            if sampling_method == 'ddpm':
                steps = backward_denoise_ddpm(model, noise, class_tensor)
            elif sampling_method == 'ddim':
                steps = backward_denoise_ddim(model, noise, class_tensor)

            generated_image = (steps[-1].to('cpu') + 1) / 2
            f = os.path.join(output_dir, f"{sampling_method}_{i + 1}.png")
            save_image(generated_image, f)

            print(f"Generated image: {f} saved to {output_dir}")


def main():
    """
    Main function to parse arguments and generate images.
    """
    args = parse_args()
    generate_images(args.model_path, args.num_images, args.class_label, args.output_dir, args.use_lora, args.sampling_method)


if __name__ == '__main__':
    main()
