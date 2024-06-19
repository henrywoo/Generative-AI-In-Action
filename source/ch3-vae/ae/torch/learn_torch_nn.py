import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the first StackedAutoencoder class with nn.Sequential
class StackedAutoencoderSequential(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 30),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.ReLU(),
            nn.Linear(100, 28 * 28),
            nn.Unflatten(1, (1, 28, 28))  # Adjusted to output (Batch, 1, 28, 28)
        )

    def forward(self, x):
        # Encode
        x_flattened = self.encoder[0](x)
        print("Sequential - Flattened:", x_flattened.shape)

        x = self.encoder[1](x_flattened)
        x = self.encoder[2](x)
        print("Sequential - After first Linear and ReLU:", x.shape)

        x = self.encoder[3](x)
        x_encoded = self.encoder[4](x)
        print("Sequential - After second Linear and ReLU (Encoded):", x_encoded.shape)

        # Decode
        x = self.decoder[0](x_encoded)
        x = self.decoder[1](x)
        print("Sequential - After first Linear and ReLU (Decoded):", x.shape)

        x = self.decoder[2](x)
        x_decoded = self.decoder[3](x)
        print("Sequential - After second Linear and Unflatten (Decoded):", x_decoded.shape)

        return x_decoded


# Define the second StackedAutoencoder class with manual layers
class StackedAutoencoderManual(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = nn.Linear(784, 100)
        self.e2 = nn.Linear(100, 30)

        # Decoder
        self.d1 = nn.Linear(30, 100)
        self.d2 = nn.Linear(100, 784)

    def forward(self, x):
        x = x.reshape(-1, 784)
        print("Manual - Flattened:", x.shape)

        x = F.relu(self.e1(x))
        print("Manual - After first Linear and ReLU:", x.shape)

        x = F.relu(self.e2(x))
        print("Manual - After second Linear and ReLU (Encoded):", x.shape)

        x = F.relu(self.d1(x))
        print("Manual - After first Linear and ReLU (Decoded):", x.shape)

        x = self.d2(x)
        x = x.reshape(-1, 1, 28, 28)
        print("Manual - After second Linear and Reshape (Decoded):", x.shape)

        return x


# Create the models
model_sequential = StackedAutoencoderSequential()
model_manual = StackedAutoencoderManual()

# Create a dummy input image
dummy_input = torch.randn(1, 1, 28, 28)

# Forward pass with debug prints for the first model
print("Debugging Sequential Model:")
output_sequential = model_sequential(dummy_input)
print("\n")

# Forward pass with debug prints for the second model
print("Debugging Manual Model:")
output_manual = model_manual(dummy_input)

# Compare the outputs
assert torch.allclose(output_manual, output_sequential, atol=1e-6), "The outputs are not the same!"
print("The outputs of both models are close enough.")
