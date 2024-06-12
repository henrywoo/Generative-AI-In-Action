import torch
import torch.nn as nn
import torch.nn.functional as F

# Simplified VQ-VAE components
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        flat_x = x.view(-1, self.embedding_dim)
        distances = (
            flat_x.pow(2).sum(1, keepdim=True)
            + self.embeddings.weight.pow(2).sum(0)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(x.shape)

        # Calculate losses
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        return quantized, commitment_loss, codebook_loss


# Example usage
vq = VectorQuantizer(num_embeddings=16, embedding_dim=32)
x = torch.randn(8, 16, 32)  # Batch of 8 inputs

quantized, commitment_loss, codebook_loss = vq(x)

print("Commitment Loss:", commitment_loss.item())
print("Codebook Loss:", codebook_loss.item())
