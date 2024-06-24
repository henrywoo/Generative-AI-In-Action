import torch
from torch import nn
import matplotlib.pyplot as plt
from dataset import train_dataset, tensor_to_pil
from config import *
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from conv_block import ConvBlock


class UNet(nn.Module):
    def __init__(self, img_channel, channels=[64, 128, 256, 512, 1024], time_emb_size=256, qsize=16, vsize=16, fsize=32,
                 cls_emb_size=32):
        super().__init__()

        channels = [img_channel] + channels

        # Time embedding
        self.time_emb = nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size, time_emb_size),
            nn.ReLU(),
        )

        # Class embedding
        self.cls_emb = nn.Embedding(10, cls_emb_size)

        # Encoder conv blocks with increasing channels
        self.enc_convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.enc_convs.append(
                ConvBlock(channels[i], channels[i + 1], time_emb_size, qsize, vsize, fsize, cls_emb_size))

        # Max pooling layers for downsampling
        self.maxpools = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # Decoder deconv layers for upsampling
        self.deconvs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i - 1], channels[-i - 2], kernel_size=2, stride=2))

        # Decoder conv blocks with decreasing channels
        self.dec_convs = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.dec_convs.append(
                ConvBlock(channels[-i - 1], channels[-i - 2], time_emb_size, qsize, vsize, fsize, cls_emb_size))

        # Output layer
        self.output = nn.Conv2d(channels[1], img_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t, cls):
        # Time embedding
        t_emb = self.time_emb(t)

        # Class embedding
        cls_emb = self.cls_emb(cls)

        # Encoder stage
        residual = []
        for i, conv in enumerate(self.enc_convs):
            x = conv(x, t_emb, cls_emb)
            if i != len(self.enc_convs) - 1:
                residual.append(x)
                x = self.maxpools[i](x)

        # Decoder stage
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            residual_x = residual.pop(-1)
            x = self.dec_convs[i](torch.cat((residual_x, x), dim=1), t_emb, cls_emb)

        return self.output(x)


def display_images(images, idx=0):
    """
    Display the noisy image, predicted noise, and generated image side by side.

    Args:
        noisy_image: Tensor of the noisy image.
        predicted_noise: Tensor of the predicted noise.
        generated_image: Tensor of the generated image.
    """
    for j in range(len(images[0])):
        plt.figure(figsize=(9, 3))
        titles = ["batch_x_t", "batch_predict_noise_t", "generated_images"]
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(titles[i])
            plt.imshow(tensor_to_pil((images[i][j] + 1) / 2))
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    from hiq import deterministic

    img = torch.stack((train_dataset[0][0], train_dataset[1][0]), dim=0).to(DEVICE)  # Combine 2 images into a batch, (2,1,48,48)
    batch_x = img * 2 - 1  # Adjust pixel values to [-1,1] to match Gaussian noise range
    batch_cls = torch.tensor([train_dataset[0][1], train_dataset[1][1]], dtype=torch.long).to(DEVICE)  # Class IDs

    batch_t = torch.randint(0, T, size=(batch_x.size(0),)).to(DEVICE)  # Randomly generate diffusion steps for each image
    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

    print("batch_cls:", batch_cls.detach().cpu().numpy())
    print("batch_t:", batch_t.detach().cpu().numpy())
    print('batch_x_t:', batch_x_t.size())
    print('batch_noise_t:', batch_noise_t.size())

    unet = UNet(img_channel=1).to(DEVICE)
    batch_predict_noise_t = unet(batch_x_t, batch_t, batch_cls)
    print('batch_predict_noise_t:', batch_predict_noise_t.size())

    # Generate images by subtracting predicted noise from the noisy images
    generated_images = batch_x_t - batch_predict_noise_t

    # Display images
    display_images([batch_x_t, batch_predict_noise_t, generated_images])
