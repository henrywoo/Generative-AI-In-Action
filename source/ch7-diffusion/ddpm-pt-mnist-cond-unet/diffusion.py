import torch
from config import *
from dataset import train_dataset, tensor_to_pil
import matplotlib.pyplot as plt

# Forward diffusion calculation parameters
betas = torch.linspace(0.0001, 0.02, T)  # (T,)
alphas = 1 - betas  # (T,)

alphas_cumprod = torch.cumprod(alphas, dim=-1)  # Cumulative product of alpha_t (T,) [a1,a2,a3,....] -> [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]),
                                dim=-1)  # Cumulative product of alpha_t-1 (T,), [1,a1,a1*a2,a1*a2*a3,.....]
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # Variance used for denoising (T,)


# Perform forward diffusion
def forward_diffusion(batch_x, batch_t):  # batch_x: (batch,channel,width,height), batch_t: (batch_size,)
    batch_noise_t = torch.randn_like(batch_x)  # Generate Gaussian noise for each image at step t (batch,channel,width,height)
    batch_alphas_cumprod = alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0), 1, 1, 1)
    batch_x_t = torch.sqrt(batch_alphas_cumprod) * batch_x + torch.sqrt(
        1 - batch_alphas_cumprod) * batch_noise_t  # Generate noisy image at step t based on the formula
    return batch_x_t, batch_noise_t


# Reverse diffusion parameters
betas = torch.linspace(0.0001, 0.02, T)  # (T,)
alphas = 1 - betas  # (T,)


def reverse_diffusion(noise, model, t, num_steps=T):
    """
    Perform reverse diffusion to generate an image from noise using the trained model.

    Args:
        noise: Tensor of random noise representing the initial input.
        model: Trained diffusion model (UNet instance).
        t: Initial time step (scalar).
        num_steps: Number of diffusion steps (same as during training).

    Returns:
        generated_image: Tensor of the generated image.
    """
    for i in range(num_steps, 0, -1):
        # Calculate alpha and sigma
        alpha = alphas[i - 1]
        sigma = torch.sqrt(variance[i - 1])

        # Predict noise at the current step
        predicted_noise = model(noise, t.view(1, 1, 1, 1))
        predicted_noise = predicted_noise.clamp(-1, 1)  # Clamp predicted noise between -1 and 1

        # Update noise and time step
        noise = alpha * noise + (1 - alpha) * (predicted_noise + sigma * torch.randn_like(noise))
        t -= 1

    # De-normalize noise back to image pixel range
    generated_image = (noise + 1) / 2

    return generated_image


def display_images(original_images, noisy_images):
    plt.figure(figsize=(3.6, 3.4))

    for i, (original, noisy) in enumerate(zip(original_images, noisy_images)):
        plt.subplot(2, len(original_images), i + 1)
        plt.title(f'Original Image {i + 1}', fontsize=8)
        plt.imshow(tensor_to_pil(original))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, len(original_images), i + 1 + len(original_images))
        plt.title(f'Noisy Image {i + 1}', fontsize=8)
        plt.imshow(tensor_to_pil((noisy + 1) / 2))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('demo_diffusion_process.png')
    plt.show()


if __name__ == '__main__':
    from hiq import deterministic

    images = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(DEVICE)  # Combine 2 images into a batch, (2,1,48,48)

    # Adjust pixel values from [0,1] to [-1,1] to match Gaussian noise range
    batch_x = images * 2 - 1

    # Randomly generate diffusion steps for each image
    batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)
    print('batch_t:', batch_t)

    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())

    # Display original and noisy images
    display_images(images, batch_x_t)
