import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
        total_loss = self.beta * commitment_loss + codebook_loss
        quantized = x + (quantized - x).detach()
        return quantized, commitment_loss, codebook_loss, total_loss

    def get_code_indices(self, flattened_inputs):
        distances = (
                torch.sum(flattened_inputs ** 2, dim=1, keepdim=True)
                + torch.sum(self.embeddings ** 2, dim=1)
                - 2 * torch.matmul(flattened_inputs, self.embeddings.T)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


def train_step(x):
    optimizer.zero_grad()
    quantized, commitment_loss, codebook_loss, total_loss = vector_quantizer(x)
    total_loss.backward()
    optimizer.step()
    return quantized, commitment_loss, codebook_loss, total_loss


def print_array(arr):
    arr_str = str(arr).replace('\n', ' ')
    return arr_str

def plot_embeddings(embeddings_list):
    num_steps = len(embeddings_list)
    fig, axes = plt.subplots(1, num_steps, figsize=(0.55 * num_steps, 1.4))
    for step in range(num_steps):
        embeddings_np = embeddings_list[step].detach().numpy()
        ax = axes[step]  # Access the correct subplot in a single row layout
        cax = ax.imshow(embeddings_np, cmap='coolwarm', aspect='auto')
        for i in range(embeddings_np.shape[0]):
            text = ax.text(0, i, f'{embeddings_np[i, 0]:.2f}', ha='center', va='center', color='white', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'ep{step}', fontsize=8)
    plt.tight_layout()
    plt.suptitle("Codebook Changes During Training", fontsize=9)
    plt.savefig('images/codebook_changes_during_training.png')
    plt.show()

def plot_losses(commitment_losses, codebook_losses):
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.plot(commitment_losses, label='Commitment Loss', marker='o', alpha=0.5)
    plt.plot(codebook_losses, label='Codebook Loss', marker='x', alpha=0.5)
    plt.xlabel('epoch', fontsize=8)
    plt.ylabel('Loss')
    plt.title('Commitment and Codebook Losses')
    plt.legend()
    plt.savefig('images/codebook_commitment_losses.png')
    plt.show()

# Initialize the quantizer
num_embeddings = 3
embedding_dim = 1
vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)

# Example input
x = torch.tensor([[0.6], [1.8]], dtype=torch.float32)

# Training step
optimizer = optim.Adam(vector_quantizer.parameters(), lr=0.1)

embeddings_list = []
commitment_losses = []
codebook_losses = []

for i in range(9):
    # Store the embeddings before the training step
    embeddings_list.append(vector_quantizer.embeddings.clone())

    # Perform one training step
    print("Input Values: ", end='')
    print(print_array(x.detach().numpy()))
    quantized, commitment_loss, codebook_loss, total_loss = train_step(x)
    print("Quantized Values: ", end='')
    print(print_array(quantized.detach().numpy()))
    print("Commitment Loss: ", commitment_loss.item())
    print("Codebook Loss: ", codebook_loss.item())
    print("Total Loss: ", total_loss.item())

    # Store the losses
    commitment_losses.append(commitment_loss.item())
    codebook_losses.append(codebook_loss.item())
    print("*"*60)

# Store the final embeddings after all training steps
embeddings_list.append(vector_quantizer.embeddings.clone())

# Plot embeddings over training steps
plot_embeddings(embeddings_list)

# Plot losses over training steps
plot_losses(commitment_losses, codebook_losses)
