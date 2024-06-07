import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc_mean = nn.Linear(100, codings_size)
        self.fc_logvar = nn.Linear(100, codings_size)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_var = self.fc_logvar(x)
        return mean, log_var


class VariationalDecoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.fc1 = nn.Linear(codings_size, 100)
        self.fc2 = nn.Linear(100, 150)
        self.fc3 = nn.Linear(150, 28 * 28)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)
        return z.view(-1, 28, 28)  # Reshape to image format


class VariationalAutoencoder(nn.Module):
    def __init__(self, codings_size):
        super().__init__()
        self.encoder = VariationalEncoder(codings_size)
        self.decoder = VariationalDecoder(codings_size)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return self.decoder(z), mean, log_var

