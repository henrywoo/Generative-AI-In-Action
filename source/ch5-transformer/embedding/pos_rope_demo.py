import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000): # 10000
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids):
        cos = self._cos_cached[position_ids].unsqueeze(1)
        sin = self._sin_cached[position_ids].unsqueeze(1)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    m = x.shape[-1] // 2
    x1 = x[..., : m]
    x2 = x[..., m:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_ = rotate_half(q)
    q_embed = (q * cos) + (q_ * sin)
    k_ = rotate_half(k)
    k_embed = (k * cos) + (k_ * sin)
    return q_embed, k_embed


# Generate sample input data
batch_size = 1
seq_len = 6
head_dim = 4  # Ensure head_dim is even

q = torch.randn(batch_size, seq_len, head_dim)
k = torch.randn(batch_size, seq_len, head_dim)

# Instantiate the LlamaRotaryEmbedding class
rotary_emb = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len)

# Generate position ids
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# Get cos and sin embeddings
cos, sin = rotary_emb(position_ids)

# Apply ROPE to the query and key tensors
q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)


# Visualize the original and rotated embeddings for the first sequence in the batch
def plot_embeddings(original, rotated, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Select the first sequence and flatten the head dimensions for visualization
    original_2d = original[0].view(seq_len, -1).detach().numpy()
    rotated_2d = rotated[0].view(seq_len, -1).detach().numpy()
    axes[0].imshow(original_2d, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Original {title}')
    axes[1].imshow(rotated_2d, cmap='viridis', aspect='auto')
    axes[1].set_title(f'ROPE {title}')
    plt.show()


plot_embeddings(q, q_embed, 'Query')
plot_embeddings(k, k_embed, 'Key')


def plot_attenuation(q_embed, k_embed):
    # Compute the dot product between q_embed and k_embed.T for each head
    attn_scores = torch.matmul(q_embed, k_embed.transpose(-1, -2))

    # Average over heads and batches
    attn_scores = attn_scores.mean(dim=[0, 1])
    attn_scores = attn_scores.detach().numpy()

    # Plot attenuation matrix
    plt.imshow(attn_scores, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('RoPE Attenuation Matrix')
    plt.show()

plot_attenuation(q_embed, k_embed)