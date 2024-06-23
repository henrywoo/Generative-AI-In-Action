import torch
import torch.nn as nn
import torch.optim as optim

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.proj(x)
        x = self.flatten(x).transpose(1, 2)
        return x

class AdaptiveTransformerDecoderOnly(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_length, n_layers, n_heads):
        super(AdaptiveTransformerDecoderOnly, self).__init__()
        self.embedding = nn.Linear(output_dim, hidden_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_length = max_length
        self.end_token = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, memory):
        batch_size, seq_length, _ = memory.size()
        tgt = torch.zeros(batch_size, 1, memory.size(-1)).to(memory.device)
        outputs = []

        for i in range(self.max_length):
            tgt_embedded = self.embedding(tgt)
            output = self.decoder(tgt_embedded, memory)
            next_token = self.fc(output[:, -1, :])
            outputs.append(next_token.unsqueeze(1))

            # Check for end token
            if torch.all(next_token == self.end_token):
                break

            tgt = torch.cat((tgt, next_token.unsqueeze(1)), dim=1)

        outputs = torch.cat(outputs, dim=1)
        return outputs

class PatchDecoderModel(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, hidden_dim, output_dim, max_length, n_layers, n_heads):
        super(PatchDecoderModel, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.decoder = AdaptiveTransformerDecoderOnly(hidden_dim, output_dim, max_length, n_layers, n_heads)

    def forward(self, x):
        memory = self.patch_embedding(x)
        output = self.decoder(memory)
        return output

# Example usage
patch_size = 4  # Dimension of each patch (4x4 for example)
in_channels = 3  # Number of input channels (e.g., RGB)
embed_dim = 256  # Dimension of embedded patches
hidden_dim = 512  # Dimension of hidden representations
output_dim = embed_dim  # Dimension of output tokens, same as embed_dim for reconstruction
max_length = 16  # Maximum length of output sequence
n_layers = 6  # Number of layers in Transformer
n_heads = 8  # Number of attention heads

model = PatchDecoderModel(patch_size, in_channels, embed_dim, hidden_dim, output_dim, max_length, n_layers, n_heads)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Dummy input for illustration
src = torch.randn(10, 3, 32, 32)  # (batch_size, in_channels, height, width)

# Forward pass
output = model(src)
