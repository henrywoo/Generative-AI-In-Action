import torch
import torch.nn as nn
import torch.nn.functional as F

class TiTok(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_chans=3, dim=1024, depth=6, heads=16, mlp_dim=2048, K=32, B=64,
                 codebook=None):
        super(TiTok, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.K = K
        self.B = B
        self.P = (image_size // patch_size) ** 2
        self.in_chans = in_chans

        # Patch embedding
        self.projection = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.encoder_pos_embedding = nn.Parameter(torch.randn(self.P + K, dim))
        self.decoder_pos_embedding = nn.Parameter(torch.randn(self.P + K, dim))

        self.e1 = nn.Linear(256, 128) # 256 is SeqLen
        self.e2 = nn.Linear(128, 64)  # 256 is SeqLen
        self.e3 = nn.Linear(64, K)  # 256 is SeqLen

        self.d3 = nn.Linear(K, 64)  # 256 is SeqLen
        self.d2 = nn.Linear(64, 128)  # 256 is SeqLen
        self.d1 = nn.Linear(128, patch_size * patch_size * in_chans)  # 256 is SeqLen
        self.latent_tokens = None

    def forward(self, x):
        B, C, H0, W0 = x.shape
        assert C == self.in_chans
        if B < self.B:
            return None, None, None
        # Patch Embedding
        p = self.projection(x)
        z = p.flatten(2)
        # Encoder
        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))

        latent_representation = z
        d = F.relu(self.d3(z))
        d = F.relu(self.d2(d))
        d = self.d1(d)
        d = d.reshape(B, C, H0, W0)
        return d, latent_representation