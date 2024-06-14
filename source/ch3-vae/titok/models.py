import torch
import torch.nn as nn
from vit_pytorch import ViT
from torchvision import transforms
from PIL import Image
import requests


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, max_seq_len):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len + 1, dim))  # max_seq_len + 1 for cls token

    def forward(self, x):
        if x.dim() == 4:  # Input is an image tensor (B, C, H, W)
            B, C, H, W = x.shape
            x = self.projection(x).flatten(2).transpose(1, 2)
        elif x.dim() == 3:  # Input is a token tensor (B, N, D)
            B, N, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:x.size(1), :]
        return x


class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=32, num_classes=1000, dim=1024, depth=6, heads=16, mlp_dim=2048,
                 num_tokens=32):
        super(TiTok, self).__init__()
        self.num_tokens = num_tokens
        max_seq_len = (image_size // patch_size) ** 2 * 2  # Max length of sequence for positional embedding
        # Encoder
        self.encoder_patch_embedding = PatchEmbedding(image_size, patch_size, dim, max_seq_len)
        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )
        # Quantizer
        self.codebook_dim = dim
        self.num_codes = 8192
        self.codebook = nn.Embedding(self.num_codes, self.codebook_dim)

        # Decoder
        self.decoder_patch_embedding = PatchEmbedding(image_size, patch_size, dim, max_seq_len)
        self.decoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, z):
        # Encoding
        x = self.encoder_patch_embedding(z)
        tokens = self.encoder.transformer(x)

        # Quantization
        tokens = tokens[:, 1:, :]  # Remove cls token
        #tokens = tokens.view(-1, tokens.size(-1))  # Flatten tokens
        tokens = tokens.reshape(-1, tokens.size(-1))
        distances = torch.cdist(tokens, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        logits = distances.view(x.size(0), -1, self.num_codes)
        quantized_tokens = self.codebook(indices)
        quantized_tokens = quantized_tokens.view(x.size(0), -1, quantized_tokens.size(-1))

        # Decoding
        mask_tokens = torch.zeros_like(quantized_tokens)
        dec_input = torch.cat((quantized_tokens, mask_tokens), dim=1)
        dec_input = self.decoder_patch_embedding(dec_input)
        reconstructed = self.decoder.transformer(dec_input)

        return logits, indices


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # Add batch dimension


if __name__ == '__main__':
    # Load the image
    image_path = 'bubble.jpg'
    input_image = load_image(image_path)

    # Instantiate the model
    model = TiTok()

    # Perform inference
    print(input_image.shape)
    output = model(input_image)
    print(output.shape)
