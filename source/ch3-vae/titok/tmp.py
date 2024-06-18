import torch
import torch.nn as nn

# Create a sample tensor with shape (Batch, Len, HiddenSize) which is (64, 32, 256)
latent_representation = torch.randn(64, 32, 256)

# Define the Linear layer to convert each vector of size Len (32) to a vector of size H * W (16 * 16 = 256)
linear_layer = nn.Linear(32, 16 * 16)

# Apply the Linear layer to the latent representation
# Since nn.Linear expects the last dimension to be the input size, we need to transpose
rr = latent_representation.transpose(1, 2)  # Shape: (64, 256, 32)
ff = linear_layer(rr)  # Shape: (64, 256, 256)

# Reshape to (Batch, HiddenSize, H, W)
final_representation = ff.view(64, 256, 16, 16)

print(final_representation.shape)  # Output: torch.Size([64, 256, 16, 16])
