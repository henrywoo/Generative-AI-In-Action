# incomplete, so tedious to convert

fp = "./checkpoints/tokenizer_imagenet256_checkpoint"

import flax
# import flax.linen as nn
import torch.nn as nn
import jax.numpy as jnp
import torch
import tensorflow as tf
from ml_collections import ConfigDict


# from vq import VQVAE


# Load Flax checkpoint
def restore_from_path(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        state = flax.serialization.from_bytes(None, f.read())
    return state


flax_params = restore_from_path(fp)


# Define the config
def get_config():
    from maskgit.configs import base_config
    import ml_collections

    config = base_config.get_config()
    config.model_class = "vqgan"
    config.d_step_per_g_step = 1
    config.image_size = 256
    config.batch_size = 256
    config.eval_batch_size = 128
    config.eval_every_steps = 10_000
    config.num_train_steps = 1_000_000
    config.pretrained_image_model = True
    config.perceptual_loss_weight = 0.1
    config.perceptual_loss_on_logit = False
    config.eval_exact_match = True
    config.eval_num = 50_000

    config.optimizer.lr = None
    config.optimizer.beta1 = 0.0
    config.optimizer.beta2 = 0.99
    config.optimizer.g_lr = 0.0001
    config.optimizer.d_lr = 0.0001

    config.vqgan = ml_collections.ConfigDict()
    config.vqgan.loss_type = "non-saturating"
    config.vqgan.g_adversarial_loss_weight = 0.1
    config.vqgan.gradient_penalty = "r1"
    config.vqgan.grad_penalty_cost = 10.0

    config.vqvae = ml_collections.ConfigDict()
    config.vqvae.quantizer = "vq"
    config.vqvae.codebook_size = 1024

    config.vqvae.entropy_loss_ratio = 0.1
    config.vqvae.entropy_temperature = 0.01
    config.vqvae.entropy_loss_type = "softmax"
    config.vqvae.commitment_cost = 0.25

    config.vqvae.filters = 128
    config.vqvae.num_res_blocks = 2
    config.vqvae.channel_multipliers = [1, 1, 2, 2, 4]
    config.vqvae.embedding_dim = 256
    config.vqvae.conv_downsample = False
    config.vqvae.activation_fn = "swish"
    config.vqvae.norm_type = "GN"

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.channel_multiplier = 1
    config.discriminator.blur_resample = True

    config.tau_anneal = ml_collections.ConfigDict()
    config.tau_anneal.tau_max = 1.0
    config.tau_anneal.tau_min = 0.6
    config.tau_anneal.tau_warmup_steps = 0
    config.tau_anneal.tau_decay_steps = 100_000
    return config


config = get_config()

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, filters, norm_layer, conv_layer, activation_fn=nn.ReLU(), use_conv_shortcut=False):
        super().__init__()
        self.norm_layer1 = norm_layer(in_channels)
        self.activation_fn = activation_fn
        self.conv1 = conv_layer(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.norm_layer2 = norm_layer(filters)
        self.conv2 = conv_layer(filters, filters, kernel_size=3, padding=1, bias=False)
        self.use_conv_shortcut = use_conv_shortcut

        if in_channels != filters:
            if use_conv_shortcut:
                self.shortcut_conv = conv_layer(filters, filters, kernel_size=3, padding=1, bias=False)
            else:
                self.shortcut_conv = conv_layer(in_channels, filters, kernel_size=1, bias=False)
        else:
            self.shortcut_conv = None

    def forward(self, x):
        residual = x
        x = self.norm_layer1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm_layer2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
        return x + residual


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.filters = config.vqvae.filters
        self.embedding_dim = config.vqvae.embedding_dim
        self.norm_type = config.vqvae.norm_type
        self.activation_fn = nn.ReLU() if config.vqvae.activation_fn == "relu" else nn.SiLU()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)

        self.res_blocks = nn.ModuleList([
            ResBlock(128, 128, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(128, 128, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(128, 128, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(128, 128, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(128, 256, self.get_norm_layer(), nn.Conv2d, self.activation_fn, use_conv_shortcut=True),
            ResBlock(256, 256, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(256, 256, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(256, 256, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(256, 512, self.get_norm_layer(), nn.Conv2d, self.activation_fn, use_conv_shortcut=True),
            ResBlock(512, 512, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(512, 512, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
            ResBlock(512, 512, self.get_norm_layer(), nn.Conv2d, self.activation_fn),
        ])

        self.final_norm = self.get_norm_layer()(512)
        self.final_conv = nn.Conv2d(512, self.embedding_dim, kernel_size=1)

    def get_norm_layer(self):
        if self.norm_type == "batch":
            return nn.BatchNorm2d
        elif self.norm_type == "layer":
            return nn.LayerNorm
        elif self.norm_type == "GN":
            return lambda num_features: nn.GroupNorm(num_groups=32, num_channels=num_features)
        else:
            raise NotImplementedError
    def forward(self, x):
        x = self.conv1(x)
        for layer in self.res_blocks:
            x = layer(x)
        x = self.final_norm(x)
        x = self.activation_fn(x)
        x = self.final_conv(x)
        return x


# Define Decoder class
class Decoder(nn.Module):
    def __init__(self, config, output_dim=3):
        super().__init__()
        self.config = config
        self.embedding_dim = config.vqvae.embedding_dim
        self.filters = config.vqvae.filters
        self.num_res_blocks = config.vqvae.num_res_blocks
        self.channel_multipliers = config.vqvae.channel_multipliers
        self.norm_layer = self.get_norm_layer(config.vqvae.norm_type)
        self.activation_fn = nn.ReLU if config.vqvae.activation_fn == "relu" else nn.SiLU

        self.conv1 = nn.Conv2d(self.embedding_dim, self.filters * self.channel_multipliers[-1], kernel_size=3,
                               padding=1)

        self.res_blocks = []
        in_channels = self.filters * self.channel_multipliers[-1]
        for i in reversed(range(len(self.channel_multipliers))):
            out_channels = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_blocks):
                self.res_blocks.append(
                    ResBlock(in_channels, out_channels, self.norm_layer, nn.Conv2d, self.activation_fn()))
                in_channels = out_channels
            if i > 0:
                self.res_blocks.append(nn.Upsample(scale_factor=2, mode='nearest'))
                self.res_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.res_blocks = nn.Sequential(*self.res_blocks)

        self.final_norm = self.norm_layer(out_channels)
        self.final_conv = nn.Conv2d(out_channels, output_dim, kernel_size=3, padding=1)

    def get_norm_layer(self, norm_type):
        if norm_type == "batch":
            return nn.BatchNorm2d
        elif norm_type == "layer":
            return nn.LayerNorm
        elif norm_type == "GN":
            return lambda num_features: nn.GroupNorm(num_groups=32, num_channels=num_features)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.final_norm(x)
        x = self.activation_fn()(x)
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


# Initialize PyTorch model
torch_model = VQVAE(config)


def print_dict_structure(d, indent=0):
    """Recursively prints the structure of a nested dictionary."""
    print("")
    for key, value in d.items():
        print(' ' * indent + str(key), end="\t")
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        elif hasattr(value, 'shape'):
            print(value.shape)


# Use the function to print the structure of flax_params
print_dict_structure(flax_params)

# Convert Flax parameters to PyTorch
def convert_params(flax_params, torch_model):
    with torch.no_grad():
        # Convert encoder params
        encoder_params = flax_params['params']['encoder']
        for i, layer in enumerate(torch_model.encoder.res_blocks):
            if isinstance(layer, ResBlock):
                flax_layer = encoder_params[f'ResBlock_{i}']
                print(f"ResBlock_{i} Conv_0 Flax shape: {flax_layer['Conv_0']['kernel'].shape}")
                print(f"ResBlock_{i} Conv_1 Flax shape: {flax_layer['Conv_1']['kernel'].shape}")
                print(f"ResBlock_{i} Conv_0 PyTorch shape: {layer.conv1.weight.shape}")
                print(f"ResBlock_{i} Conv_1 PyTorch shape: {layer.conv2.weight.shape}")
                layer.conv1.weight.copy_(torch.tensor(flax_layer['Conv_0']['kernel']).permute(3, 2, 0, 1))
                layer.conv2.weight.copy_(torch.tensor(flax_layer['Conv_1']['kernel']).permute(3, 2, 0, 1))
                if 'Conv_2' in flax_layer:
                    print(f"ResBlock_{i} Conv_2 Flax shape: {flax_layer['Conv_2']['kernel'].shape}")
                    print(f"ResBlock_{i} Conv_2 PyTorch shape: {layer.shortcut_conv.weight.shape}")
                    layer.shortcut_conv.weight.copy_(torch.tensor(flax_layer['Conv_2']['kernel']).permute(3, 2, 0, 1))

        # Convert initial and final conv layers of encoder
        print(f"Conv_0 Flax shape: {flax_params['params']['encoder']['Conv_0']['kernel'].shape}")
        print(f"Conv_0 PyTorch shape: {torch_model.encoder.conv1.weight.shape}")
        torch_model.encoder.conv1.weight.copy_(torch.tensor(flax_params['params']['encoder']['Conv_0']['kernel']).permute(3, 2, 0, 1))

        print(f"Conv_1 Flax shape: {flax_params['params']['encoder']['Conv_1']['kernel'].shape}")
        print(f"Conv_1 PyTorch shape: {torch_model.encoder.final_conv.weight.shape}")
        torch_model.encoder.final_conv.weight.copy_(torch.tensor(flax_params['params']['encoder']['Conv_1']['kernel']).permute(3, 2, 0, 1))

        # Convert decoder params
        decoder_params = flax_params['params']['decoder']
        for i, layer in enumerate(torch_model.decoder.res_blocks):
            if isinstance(layer, ResBlock):
                flax_layer = decoder_params[f'ResBlock_{i}']
                print(f"ResBlock_{i} Conv_0 Flax shape: {flax_layer['Conv_0']['kernel'].shape}")
                print(f"ResBlock_{i} Conv_1 Flax shape: {flax_layer['Conv_1']['kernel'].shape}")
                print(f"ResBlock_{i} Conv_0 PyTorch shape: {layer.conv1.weight.shape}")
                print(f"ResBlock_{i} Conv_1 PyTorch shape: {layer.conv2.weight.shape}")
                layer.conv1.weight.copy_(torch.tensor(flax_layer['Conv_0']['kernel']).permute(3, 2, 0, 1))
                layer.conv2.weight.copy_(torch.tensor(flax_layer['Conv_1']['kernel']).permute(3, 2, 0, 1))
                if 'Conv_2' in flax_layer:
                    print(f"ResBlock_{i} Conv_2 Flax shape: {flax_layer['Conv_2']['kernel'].shape}")
                    print(f"ResBlock_{i} Conv_2 PyTorch shape: {layer.shortcut_conv.weight.shape}")
                    layer.shortcut_conv.weight.copy_(torch.tensor(flax_layer['Conv_2']['kernel']).permute(3, 2, 0, 1))

        # Convert initial and final conv layers of decoder
        print(f"Conv_0 Flax shape: {flax_params['params']['decoder']['Conv_0']['kernel'].shape}")
        print(f"Conv_0 PyTorch shape: {torch_model.decoder.conv1.weight.shape}")
        torch_model.decoder.conv1.weight.copy_(torch.tensor(flax_params['params']['decoder']['Conv_0']['kernel']).permute(3, 2, 0, 1))

        print(f"Conv_1 Flax shape: {flax_params['params']['decoder']['Conv_1']['kernel'].shape}")
        print(f"Conv_1 PyTorch shape: {torch_model.decoder.final_conv.weight.shape}")
        torch_model.decoder.final_conv.weight.copy_(torch.tensor(flax_params['params']['decoder']['Conv_1']['kernel']).permute(3, 2, 0, 1))

        # Convert vector quantizer params
        torch_model.quantizer.codebook.weight.copy_(torch.tensor(flax_params['params']['quantizer']['codebook']))

convert_params(flax_params, torch_model)


# Verify the conversion
input_data = torch.randn(1, 3, 256, 256)
torch_output, _ = torch_model(input_data)
print(torch_output.shape)
