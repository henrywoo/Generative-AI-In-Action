import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hiq.cv_torch import get_cv_dataset, DS_PATH_MNIST

# 基本参数
batch_size = 30000
original_dim = 784
latent_dim = 3
intermediate_dim = 256
epochs = 50
kappa = 20
epsilon = 1e-7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
def load_data(data_path, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    loader_params = dict(
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )
    dataloader = get_cv_dataset(path=str(data_path),
                                batch_size=batch_size,
                                num_workers=8,
                                transform=transform,
                                image_size=None,
                                return_type="pair",
                                return_loader=True,
                                convert_rgb=False,
                                **loader_params)
    return dataloader['train'], dataloader['test']

train_loader, test_loader = load_data(DS_PATH_MNIST, batch_size)

# 定义模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = F.normalize(h, p=2, dim=-1)
        return mu

    def reparameterize(self, mu):
        dims = mu.size(-1)
        x = np.arange(-1 + epsilon, 1, epsilon)
        y = kappa * x + np.log(1 - x ** 2) * (dims - 3) / 2
        y = np.cumsum(np.exp(y - y.max()))
        y = y / y[-1]
        W = torch.tensor(np.interp(np.random.random(10 ** 6), y, x), dtype=torch.float32, device=mu.device)
        idx = torch.randint(0, 10 ** 6, (mu.size(0), 1), dtype=torch.long, device=mu.device)
        w = W[idx]
        eps = torch.randn_like(mu)
        nu = eps - (eps * mu).sum(dim=1, keepdim=True) * mu
        nu = F.normalize(nu, p=2, dim=-1)
        return w * mu + torch.sqrt(1 - w ** 2) * nu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu = self.encode(x.view(-1, original_dim))
        z = self.reparameterize(mu)
        return self.decode(z), mu

# 损失函数
def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, original_dim), reduction='sum')
    return BCE

# 训练模型
def train(model, epoch, train_loader, optimizer, train_loss_history):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False)):
        data = data.view(-1, original_dim).to(device)
        optimizer.zero_grad()
        recon_batch, mu = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_loss_history.append(avg_train_loss)
    print(f'====> Epoch: {epoch} Average train loss: {avg_train_loss:.4f}')

def validate(model, test_loader, val_loss_history):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, original_dim).to(device)
            recon_batch, mu = model(data)
            test_loss += loss_function(recon_batch, data).item()
    avg_val_loss = test_loss / len(test_loader.dataset)
    val_loss_history.append(avg_val_loss)
    print(f'====> Test set loss: {avg_val_loss:.4f}')

if __name__ == '__main__':
    # 训练模型
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loss_history = []
    val_loss_history = []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train(model, epoch, train_loader, optimizer, train_loss_history)
        validate(model, test_loader, val_loss_history)

    # 可视化生成的结果
    with torch.no_grad():
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        for i in range(n):
            for j in range(n):
                # 在hyperball球面上取点
                z_sample = torch.randn(1, latent_dim, device=device)
                z_sample /= z_sample.norm()
                x_decoded = model.decode(z_sample).view(digit_size, digit_size).cpu()
                digit = x_decoded.numpy()
                figure[i * digit_size:(i + 1) * digit_size, j * digit_size:(j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('test.png')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()
