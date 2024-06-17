import torch
from config import *
from dataset import train_dataset, tensor_to_pil
import matplotlib.pyplot as plt

# 前向diffusion计算参数
betas = torch.linspace(0.0001, 0.02, T)  # (T,)
alphas = 1 - betas  # (T,)

alphas_cumprod = torch.cumprod(alphas, dim=-1)  # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]),
                                dim=-1)  # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
variance = (1 - alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)  # denoise用的方差   (T,)


# 执行前向加噪
def forward_diffusion(batch_x, batch_t):  # batch_x: (batch,channel,width,height), batch_t: (batch_size,)
    batch_noise_t = torch.randn_like(batch_x)  # 为每张图片生成第t步的高斯噪音   (batch,channel,width,height)
    batch_alphas_cumprod = alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0), 1, 1, 1)
    batch_x_t = torch.sqrt(batch_alphas_cumprod) * batch_x + torch.sqrt(
        1 - batch_alphas_cumprod) * batch_noise_t  # 基于公式直接生成第t步加噪后图片
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


if __name__ == '__main__':
    batch_x = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(DEVICE)  # 2个图片拼batch, (2,1,48,48)

    # 加噪前的样子
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil(batch_x[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    plt.show()

    batch_x = batch_x * 2 - 1  # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)  # 每张图片随机生成diffusion步数
    # batch_t=torch.tensor([5,100],dtype=torch.long)
    print('batch_t:', batch_t)

    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())

    # 加噪后的样子
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_pil((batch_x_t[0] + 1) / 2))
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_pil((batch_x_t[1] + 1) / 2))
    plt.show()
