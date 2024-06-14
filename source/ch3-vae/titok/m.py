import torch
import torch.nn as nn
from vit_pytorch import ViT
import torch.optim as optim

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + 1, dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_tokens = self.pos_embedding[:1].expand(B, -1, -1)
        x += self.pos_embedding[1 : x.size(1) + 1, :]
        return x


class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=16, dim=1024, depth=6, heads=16, mlp_dim=2048, K=8):
        super(TiTok, self).__init__()
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
        self.K = K
        self.latent_tokens = nn.Parameter(torch.randn(K, dim))

    def forward(self, x):
        B = x.size(0)
        patch_embeddings = self.patch_embedding(x)
        latent_tokens = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)
        combined_input = torch.cat((patch_embeddings, latent_tokens), dim=1)
        encoded = self.encoder.transformer(combined_input)
        latent_representation = encoded[:, -self.K :, :]
        return latent_representation


# Example usage
K=3
model = TiTok(dim=2, K=K)
#print(model.latent_tokens)
input_image = torch.randn(1, 3, 256, 256)
latent_representation = model(input_image)
print(model.latent_tokens)
print(latent_representation.shape)  # Should be (1, K, dim)
print(latent_representation)


