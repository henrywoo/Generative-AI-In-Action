import torch
import torch.nn as nn
from vit_pytorch import ViT


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, in_chans=3):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)
        pos_embeddings = self.pos_embedding[1: x.size(1) + 1, :]
        return x + pos_embeddings


class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=32, in_chans=3, dim=1024, depth=6, heads=16, mlp_dim=2048, K=32,
                 codebook=None):
        super(TiTok, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.K = K
        self.in_chans = in_chans
        self.codebook_dim = codebook.shape[1] if codebook is not None else dim
        self.num_codes = codebook.shape[0] if codebook is not None else 1024

        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, in_chans)

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

        self.codebook = codebook
        assert codebook is not None and codebook.shape[0] == self.num_codes and codebook.shape[1] == self.codebook_dim
        self.codebook.requires_grad = False

        self.num_patches = (self.image_size // self.patch_size) ** 2

        # TODO: I don't need the entire VIT. I just need the transformer part!
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
        self.deprojection = nn.Linear(dim, patch_size * patch_size * self.in_chans)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embedding.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.patch_embedding.pos_embedding, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def straight_through_estimator(self, x, indices):
        quantized = self.codebook[indices]
        return x + (quantized - x).detach()

    def forward(self, x):
        B = x.size(0)
        C = x.size(1)
        assert C == self.in_chans
        H0 = x.size(2)
        W0 = x.size(3)

        # Patch Embedding
        patch_embeddings = self.patch_embedding(x)
        latent_tokens = torch.zeros(B, self.K, self.dim, device=patch_embeddings.device)
        mask_tokens = torch.zeros(B, self.num_patches, self.dim, device=patch_embeddings.device)
        self.codebook = self.codebook.to(patch_embeddings.device)

        combined_input = torch.cat((patch_embeddings, latent_tokens), dim=1)

        # Encoder
        encoded = self.encoder.transformer(combined_input)
        latent_representation = encoded[:, -self.K:, :]

        # Quantize
        distances = torch.cdist(latent_representation, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized_tokens = self.straight_through_estimator(latent_representation, indices)

        # Concatenate quantized tokens with mask tokens
        dec_input = torch.cat((quantized_tokens, mask_tokens), dim=1)

        # Decoder
        decoded = self.decoder.transformer(dec_input)

        # Convert back to image pixels
        z = decoded[:, self.K:, :]
        decoded_patches = z.view(B, self.num_patches, self.dim)
        decoded_patches = self.deprojection(decoded_patches)
        decoded_patches = decoded_patches.view(B, self.num_patches, self.in_chans, self.patch_size, self.patch_size)
        decoded_patches = decoded_patches.permute(0, 2, 1, 3, 4)
        reconstructed = decoded_patches.contiguous().view(B, self.in_chans, H0, W0)
        return reconstructed, quantized_tokens, latent_representation
