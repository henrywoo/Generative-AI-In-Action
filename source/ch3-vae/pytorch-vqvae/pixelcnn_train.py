import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from pixelcnn import PixelCNN
import matplotlib.pyplot as plt
from tqdm import tqdm

file_vqvae = 'models/vqvae/best.pt'
file_codebook = 'models/vqvae/latent_codes.pt'
file_pixelcnn = 'models/vqvae/best_pixelcnn.pt'

# Load the latent codes
latent_codes = torch.load(file_codebook)

# Add a channel dimension to the latent codes
latent_codes = latent_codes.unsqueeze(1)  # Shape will be (N, 1, 7, 7)

# Create dataset and dataloader
dataset = TensorDataset(latent_codes)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Function to train the PixelCNN model
def train_pixelcnn(pixelcnn_model, dataloader, device, epochs=10, lr=1e-3):
    optimizer = optim.Adam(pixelcnn_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    pixelcnn_model.to(device)

    all_losses = []

    for epoch in range(epochs):
        pixelcnn_model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            latents = batch[0].to(device)  # (batch_size, 1, 7, 7)
            latents = latents.long()  # Ensure latents are long type for cross-entropy loss

            optimizer.zero_grad()
            output = pixelcnn_model(latents.float())  # Convert latents to float for convolutional layers
            loss = criterion(output, latents.squeeze(1))  # Remove channel dimension for target

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        average_loss = total_loss / len(dataloader)
        all_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

    # Plotting the loss curve
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(range(1, epochs + 1), all_losses, label='Loss', marker='o', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig('train_pixelcnn_loss.png')
    plt.show()


# Example usage to train the PixelCNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_embeddings = 512
embed_dim = 1  # Because latent codes now have shape (N, 1, 7, 7)

pixelcnn_model = PixelCNN(num_embeddings=num_embeddings, embed_dim=embed_dim)
train_pixelcnn(pixelcnn_model, dataloader, device, epochs=30, lr=1e-3)

# Save the trained PixelCNN model
torch.save(pixelcnn_model.state_dict(), file_pixelcnn)
