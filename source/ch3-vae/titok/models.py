import torch
import torch.nn as nn
from vit_pytorch import ViT


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)
        pos_embeddings = self.pos_embedding[1 : x.size(1) + 1, :]
        return x + pos_embeddings


class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=32, dim=1024, depth=6, heads=16, mlp_dim=2048, K=32, codebook=None):
        super(TiTok, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.K = K
        self.codebook_dim = codebook.shape[1] if codebook is not None else dim
        self.num_codes = codebook.shape[0] if codebook is not None else 1024

        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)

        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1,  # Not used for classification
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.codebook = nn.Embedding(self.num_codes, self.codebook_dim)
        if codebook is not None:
            self.codebook.weight.data.copy_(codebook)  # Use the pre-trained codebook

        self.latent_tokens = nn.Parameter(torch.randn(K, dim))
        self.mask_token = nn.Parameter(torch.randn(1, dim))

        self.decoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1,  # Not used for classification
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
        )
        self.deprojection = nn.ConvTranspose2d(dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B = x.size(0)
        C = x.size(1)
        H0 = x.size(2)
        W0 = x.size(3)

        # Patch Embedding
        patch_embeddings = self.patch_embedding(x)
        latent_tokens = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)
        combined_input = torch.cat((patch_embeddings, latent_tokens), dim=1)

        # Encoder
        encoded = self.encoder.transformer(combined_input)
        latent_representation = encoded[:, -self.K :, :]

        # Quantize
        distances = torch.cdist(latent_representation, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        quantized_tokens = self.codebook(indices)

        # Create mask tokens
        num_patches = (self.image_size // self.patch_size) ** 2
        mask_tokens = self.mask_token.expand(num_patches, B, -1).transpose(0, 1)

        # Concatenate quantized tokens with mask tokens
        dec_input = torch.cat((quantized_tokens, mask_tokens), dim=1)

        # Decoder
        decoded = self.decoder.transformer(dec_input)

        # Convert back to image pixels
        z = decoded[:, self.K :, :]
        patch_dim = self.patch_size * self.patch_size * 3
        decoded_patches = z.view(B, num_patches, self.dim)
        decoded_patches = decoded_patches.transpose(1, 2).contiguous()
        decoded_patches = decoded_patches.view(B, self.dim, H0 // self.patch_size, W0 // self.patch_size)

        # Deprojection
        reconstructed = self.deprojection(decoded_patches)
        return reconstructed, quantized_tokens
