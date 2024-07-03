import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
critic_iterations = 5
epochs = 100
latent_dim = 100
clamp_lower = -0.01
clamp_upper = 0.01
learning_rate = 0.00005

# Transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 Dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Critic Model
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 3, 32, 32)


# Initialize models
critic = Critic().cuda()
generator = Generator().cuda()

# Optimizers
critic_optimizer = optim.RMSprop(critic.parameters(), lr=learning_rate)
generator_optimizer = optim.RMSprop(generator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i, (real_data, _) in enumerate(train_loader):
        real_data = real_data.cuda()

        for _ in range(critic_iterations):
            # Train Critic
            z = torch.randn(batch_size, latent_dim).cuda()
            fake_data = generator(z).detach()

            critic_loss = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Weight clipping
            for p in critic.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

        # Train Generator
        z = torch.randn(batch_size, latent_dim).cuda()
        fake_data = generator(z)
        generator_loss = -torch.mean(critic(fake_data))

        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

    print(
        f"Epoch [{epoch + 1}/{epochs}]  Critic Loss: {critic_loss.item():.4f}  Generator Loss: {generator_loss.item():.4f}")

print("Training finished.")
