import torch
import torch.nn as nn

class FSQ(nn.Module):
    def __init__(self, levels):
        super(FSQ, self).__init__()
        self._levels = torch.tensor(levels, dtype=torch.int32)
        self._basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0)

    def forward(self, x):
        # Ensure x is quantized to the levels
        quantized = self.quantize(x)
        indices = self.calculate_indices(quantized)
        return quantized, indices

    def quantize(self, x):
        # Dummy quantization: round to nearest level
        quantized = torch.round(x)
        return quantized

    def calculate_indices(self, quantized):
        indices = (quantized * self._basis).sum(dim=-1)
        return indices

# Example usage
levels = [8, 5, 5, 5]
quantizer = FSQ(levels)

x = torch.tensor([[[3.2, 2.4, 1.1, 4.7]]])  # Shape [1, 1, 4]
xhat, indices = quantizer(x)

print("Quantized:", xhat)
print("Indices:", indices)
