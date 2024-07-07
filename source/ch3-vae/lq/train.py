import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Latent(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        self.is_continuous = False
        self.num_latents = 0
        self.num_inputs = 0

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

class QuantizedLatent(Latent):
    def __init__(self, num_latents, num_values_per_latent, optimize_values, device='cpu'):
        super().__init__()
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents
        self.optimize_values = optimize_values
        self.device = device

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        # Initialize codebooks Vj as nn.Parameter
        self._values_per_latent = nn.ParameterList([nn.Parameter(torch.linspace(-0.5, 0.5, self.num_values_per_latent[i], device=self.device)) for i in range(num_latents)])

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [v.detach() for v in self._values_per_latent]

    @staticmethod
    def quantize(x, values):
        # Ensure x has the correct shape for broadcasting
        distances = torch.abs(x.unsqueeze(-1) - values)
        index = torch.argmin(distances, dim=-1)
        return values[index], index

    def forward(self, x):
        z_continuous = x
        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(z_continuous.T, self.values_per_latent)]
        quantized = torch.stack([qi[0] for qi in quantized_and_indices]).T
        indices = torch.stack([qi[1] for qi in quantized_and_indices]).T
        quantized_sg = z_continuous + (quantized - z_continuous).detach()
        Lquantize = F.mse_loss(z_continuous.detach(), quantized)
        Lcommit = F.mse_loss(z_continuous, quantized.detach())
        outs = {
            'z_continuous': z_continuous,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices,
            'Lquantize': Lquantize,
            'Lcommit': Lcommit
        }
        return outs

    def sample(self):
        ret = []
        for values in self.values_per_latent:
            ret.append(torch.choice(values))
        return torch.tensor(ret, device=self.device)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.linear(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return torch.sigmoid(self.linear(z))

def train_qlae(dataset, batch_size, alpha, beta1, beta2, weight_decay, lambda_reconstruct, lambda_quantize, lambda_commit, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = dataset.shape[1]  # Assuming dataset is a tensor of shape [N, D]
    latent_dim = 10  # Example latent dimension
    num_values_per_latent = 5

    encoder = Encoder(input_dim, latent_dim).to(device)
    decoder = Decoder(latent_dim, input_dim).to(device)
    quantized_latent = QuantizedLatent(latent_dim, num_values_per_latent, optimize_values=True, device=device).to(device)

    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=alpha, betas=(beta1, beta2), weight_decay=weight_decay)
    optimizer_v = optim.Adam(quantized_latent.parameters(), lr=alpha, betas=(beta1, beta2))

    for epoch in range(num_epochs):
        for i in range(0, len(dataset), batch_size):
            x = dataset[i:i+batch_size].to(device)

            # Forward pass through encoder and quantized latent
            z_continuous = encoder(x)
            quantized_outputs = quantized_latent(z_continuous)
            z = quantized_outputs['z_hat']
            Lquantize = quantized_outputs['Lquantize']
            Lcommit = quantized_outputs['Lcommit']

            # Reconstruction loss
            x_reconstructed = decoder(z)
            Lreconstruct = F.binary_cross_entropy(x_reconstructed, x)

            # Total loss
            Lqlae = lambda_reconstruct * Lreconstruct + lambda_quantize * Lquantize + lambda_commit * Lcommit

            # Backward pass and optimization
            optimizer.zero_grad()
            optimizer_v.zero_grad()
            Lqlae.backward()
            optimizer.step()
            optimizer_v.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {Lqlae.item():.4f}")

# Example usage
if __name__ == "__main__":
    dataset = torch.randn(100, 784)  # Example dataset with 100 samples, each of dimension 784
    train_qlae(dataset, batch_size=32, alpha=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01,
               lambda_reconstruct=1.0, lambda_quantize=1.0, lambda_commit=1.0, num_epochs=10)
