import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        initial_embeddings = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).view(num_embeddings, embedding_dim)
        self.embeddings = nn.Parameter(initial_embeddings)

    def forward(self, x):
        input_shape = x.shape
        flattened = x.view(-1, self.embedding_dim)
        encoding_indices = self.get_code_indices(flattened)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embeddings)
        quantized = quantized.view(input_shape)
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = self.beta * commitment_loss + codebook_loss
        return quantized, loss

    def get_code_indices(self, flattened_inputs):
        distances = (
            torch.sum(flattened_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flattened_inputs, self.embeddings.T)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

# Initialize the quantizer
num_embeddings = 3
embedding_dim = 1
vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

# Example input
x = torch.tensor([[0.6], [1.4]], dtype=torch.float32)

# Training step
optimizer = optim.Adam(vector_quantizer.parameters(), lr=0.1)

def train_step(x):
    optimizer.zero_grad()
    quantized, loss = vector_quantizer(x)
    loss.backward()
    optimizer.step()
    return quantized, loss

def print_array(arr):
    arr_str = str(arr).replace('\n', ' ')
    return arr_str

print("Original Embeddings: ", end='')
print(print_array(vector_quantizer.embeddings.detach().numpy()))

# Perform one training step
print("Input Values: ", end='')
print(print_array(x.detach().numpy()))
quantized, loss = train_step(x)
print("Quantized Values: ", end='')
print(print_array(quantized.detach().numpy()))
print("Loss: ", end='')
print(loss.item())

# Show updated embeddings after training step
print("Updated Embeddings: ", end='')
print(print_array(vector_quantizer.embeddings.detach().numpy()))
