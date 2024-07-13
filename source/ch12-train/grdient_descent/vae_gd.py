import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def compute_fisher_information(model, data_loader):
    fisher_information = {}
    model.eval()
    for param in model.parameters():
        fisher_information[param] = torch.zeros_like(param)

    for x, _ in data_loader:
        x = x.view(-1, 784)
        model.zero_grad()
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar)
        loss.backward()

        for param in model.parameters():
            fisher_information[param] += param.grad ** 2

    for param in fisher_information:
        fisher_information[param] /= len(data_loader)

    return fisher_information

# 训练过程示例
def train_vae_with_natural_gradient(model, data_loader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        for x, _ in data_loader:
            x = x.view(-1, 784)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()

            # 使用Fisher信息矩阵的逆进行更新（自然梯度下降）
            fisher_information = compute_fisher_information(model, data_loader)
            for param in model.parameters():
                param.grad /= (fisher_information[param] + 1e-8)  # 添加一个小的常数以防止除零

            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 示例数据加载器（使用MNIST数据集）
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 初始化和训练VAE
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
train_vae_with_natural_gradient(vae, train_loader, num_epochs=10, learning_rate=1e-3)
