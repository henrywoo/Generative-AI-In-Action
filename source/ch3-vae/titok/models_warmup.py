import torch
import torch.nn as nn

class Detokenizer(nn.Module):
    def __init__(self, input_dim, grid_size, B):
        super().__init__()
        self.linear = nn.Linear(input_dim, grid_size * grid_size)
        self.grid_size = grid_size
        self.B = B

    def forward(self, x):
        r = x.transpose(1, 2)  # Shape: (64, 256, 32)
        f = self.linear(r)  # Shape: (64, 256, 256)
        return f.view(self.B, -1, self.grid_size, self.grid_size) # -1 should be 256, HD



class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=32, in_chans=3, dim=1024, depth=6, heads=16, mlp_dim=2048, K=32, B=64,
                 codebook=None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.K = K
        self.B = B
        self.P = (image_size // patch_size) ** 2
        self.in_chans = in_chans
        self.codebook_dim = codebook.shape[1] if codebook is not None else dim
        self.num_codes = codebook.shape[0] if codebook is not None else 1024

        # Patch embedding
        self.projection = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.encoder_pos_embedding = nn.Parameter(torch.randn(self.P + K, dim))
        self.decoder_pos_embedding = nn.Parameter(torch.randn(self.P + K, dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        self.deprojection = nn.Linear(dim, patch_size * patch_size * self.in_chans)

        # Latent and mask tokens
        self.latent_tokens = nn.Parameter(torch.randn(B, K, dim))
        self.mask_tokens = nn.Parameter(torch.randn(B, self.P, dim))

        self.codebook = codebook
        assert codebook is not None and codebook.shape[0] == self.num_codes and codebook.shape[1] == self.codebook_dim
        self.codebook.requires_grad = False
        self.detokenizer = Detokenizer(input_dim=K, grid_size=16, B=self.B)

    def straight_through_estimator(self, x, indices):
        quantized = self.codebook[indices]
        return x + (quantized - x).detach()

    def reconstruct(self, x):
        B, C, H0, W0 = x.shape
        assert C == self.in_chans
        if B < self.B:
            return None, None, None
        # Patch Embedding
        patch_embeddings = self.projection(x).flatten(2).transpose(1, 2)
        patch_embeddings = torch.cat((patch_embeddings, self.latent_tokens), dim=1)
        combined_input = patch_embeddings + self.encoder_pos_embedding
        # Encoder
        encoded = self.encoder(combined_input)
        latent_representation = encoded[:, -self.K:, :]
        # Quantize
        self.codebook = self.codebook.to(patch_embeddings.device)
        distances = torch.cdist(latent_representation, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized_tokens = self.straight_through_estimator(latent_representation, indices)
        # Concatenate quantized tokens with mask tokens
        dec_input = torch.cat((quantized_tokens, self.mask_tokens), dim=1)
        dec_input = dec_input + self.decoder_pos_embedding
        # Decoder
        decoded = self.decoder(dec_input)
        # Convert back to image pixels
        z = decoded[:, self.K:, :]
        decoded_patches = z.view(B, -1, self.dim)
        decoded_patches = self.deprojection(decoded_patches)
        decoded_patches = decoded_patches.view(B, self.P, self.in_chans, self.patch_size, self.patch_size)
        decoded_patches = decoded_patches.permute(0, 2, 1, 3, 4)
        reconstructed = decoded_patches.contiguous().view(B, self.in_chans, H0, W0)
        return reconstructed, quantized_tokens, latent_representation

    def warm_up(self, x):
        B, C, H0, W0 = x.shape
        assert C == self.in_chans
        if B < self.B:
            return None, None, None
        # Patch Embedding
        patch_embeddings = self.projection(x).flatten(2).transpose(1, 2)
        patch_embeddings = torch.cat((patch_embeddings, self.latent_tokens), dim=1)
        combined_input = patch_embeddings + self.encoder_pos_embedding
        # Encoder
        encoded = self.encoder(combined_input)
        latent_representation = encoded[:, -self.K:, :]
        # Quantize
        #self.codebook = self.codebook.to(patch_embeddings.device)
        #distances = torch.cdist(latent_representation, self.codebook)
        #indices = distances.argmin(dim=-1)
        #quantized_tokens = self.straight_through_estimator(latent_representation, indices)

        final_representation = self.detokenizer(latent_representation)
        return final_representation

    def forward(self, x):
        return self.warm_up(x)