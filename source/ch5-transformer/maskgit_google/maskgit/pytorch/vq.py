import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, conv_layer, activation_fn=torch.nn.ReLU, use_conv_shortcut=False):
        super().__init__()
        self.norm_layer = norm_layer(out_channels)
        self.activation_fn = activation_fn
        self.use_conv_shortcut = use_conv_shortcut

        self.conv1 = conv_layer(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = conv_layer(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_conv_shortcut:
            self.shortcut_conv = conv_layer(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.shortcut_conv = conv_layer(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm_layer(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm_layer(x)
        x = self.activation_fn(x)
        x = self.conv2(x)

        if self.use_conv_shortcut:
            residual = self.shortcut_conv(residual)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.filters = config.vqvae.filters
        self.num_res_blocks = config.vqvae.num_res_blocks
        self.channel_multipliers = config.vqvae.channel_multipliers
        self.embedding_dim = config.vqvae.embedding_dim
        self.conv_downsample = config.vqvae.conv_downsample
        self.norm_layer = self.get_norm_layer(config.vqvae.norm_type)
        self.activation_fn = nn.ReLU if config.vqvae.activation_fn == "relu" else nn.SiLU

        self.conv1 = nn.Conv2d(3, self.filters, kernel_size=3, padding=1, bias=False)

        self.res_blocks = []
        in_channels = self.filters
        for i, multiplier in enumerate(self.channel_multipliers):
            out_channels = self.filters * multiplier
            for _ in range(self.num_res_blocks):
                self.res_blocks.append(ResBlock(in_channels, out_channels, self.norm_layer, nn.Conv2d, self.activation_fn))
                in_channels = out_channels
            if i < len(self.channel_multipliers) - 1:
                if self.conv_downsample:
                    self.res_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
                else:
                    self.res_blocks.append(nn.MaxPool2d(2))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.final_norm = self.norm_layer(out_channels)
        self.final_conv = nn.Conv2d(out_channels, self.embedding_dim, kernel_size=1)

    def get_norm_layer(self, norm_type):
        if norm_type == "batch":
            return nn.BatchNorm2d
        elif norm_type == "layer":
            return nn.LayerNorm
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.final_norm(x)
        x = self.activation_fn()(x)
        x = self.final_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config, output_dim=3):
        super().__init__()
        self.config = config
        self.filters = config.vqvae.filters
        self.embedding_dim = config.vqvae.embedding_dim
        self.num_res_blocks = config.vqvae.num_res_blocks
        self.channel_multipliers = config.vqvae.channel_multipliers
        self.norm_layer = self.get_norm_layer(config.vqvae.norm_type)
        self.activation_fn = F.relu if config.vqvae.activation_fn == "relu" else F.silu

        self.conv1 = nn.Conv2d(self.embedding_dim, self.filters * self.channel_multipliers[-1], kernel_size=3, padding=1)

        self.res_blocks = []
        for i in reversed(range(len(self.channel_multipliers))):
            for _ in range(self.num_res_blocks):
                self.res_blocks.append(ResBlock(self.filters * self.channel_multipliers[i], self.norm_layer, nn.Conv2d, self.activation_fn))
            if i > 0:
                self.res_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.res_blocks.append(nn.Conv2d(self.filters * self.channel_multipliers[i], self.filters * self.channel_multipliers[i-1], kernel_size=3, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.final_norm = self.norm_layer
        self.final_conv = nn.Conv2d(self.filters, output_dim, kernel_size=3, padding=1)

    def get_norm_layer(self, norm_type):
        if norm_type == "batch":
            return nn.BatchNorm2d
        elif norm_type == "layer":
            return nn.LayerNorm
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.final_norm(x)
        x = self.activation_fn(x)
        x = self.final_conv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.codebook_size = config.vqvae.codebook_size
        self.embedding_dim = config.vqvae.embedding_dim
        self.commitment_cost = config.vqvae.commitment_cost
        self.codebook = nn.Embedding(self.codebook_size, self.embedding_dim)

    def forward(self, x):
        # Flatten input except for the last dimension
        flat_x = x.view(-1, self.embedding_dim)

        # Compute distances
        distances = (flat_x.pow(2).sum(dim=1, keepdim=True) +
                     self.codebook.weight.pow(2).sum(dim=1) -
                     2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get encoding indices and encodings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.codebook_size, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = self.quantize(encodings)

        # Compute loss
        e_latent_loss = F.mse_loss(quantized.detach(), flat_x) * self.commitment_cost
        q_latent_loss = F.mse_loss(quantized, flat_x.detach())
        loss = e_latent_loss + q_latent_loss

        # Add loss to quantized
        quantized = flat_x + (quantized - flat_x).detach()

        return quantized.view(*x.shape), loss, encoding_indices.view(*x.shape[:-1])

    def quantize(self, encodings):
        return torch.matmul(encodings, self.codebook.weight)


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantizer = VectorQuantizer(config)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, loss, _ = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, loss
