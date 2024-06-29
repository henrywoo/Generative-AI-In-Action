import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class GAN(pl.LightningModule):
    def __init__(self, noise_dim, data_dim):
        super(GAN, self).__init__()
        self.generator = Generator(noise_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.noise_dim = noise_dim
        self.automatic_optimization = False  # 关闭自动优化

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        real_imgs = batch[0]
        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.noise_dim)
        z = z.type_as(real_imgs)

        # train generator
        self.generated_imgs = self(z)
        valid = torch.ones(real_imgs.size(0), 1)
        valid = valid.type_as(real_imgs)
        g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)

        optimizer_g.zero_grad()
        self.manual_backward(g_loss)
        optimizer_g.step()

        # train discriminator
        valid = torch.ones(real_imgs.size(0), 1)
        valid = valid.type_as(real_imgs)
        fake = torch.zeros(real_imgs.size(0), 1)
        fake = fake.type_as(real_imgs)

        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizer_d.zero_grad()
        self.manual_backward(d_loss)
        optimizer_d.step()

        self.log('g_loss', g_loss, prog_bar=True)
        self.log('d_loss', d_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0.999
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d]


# 创建数据集和数据加载器
data_dim = 28 * 28
noise_dim = 100
x = torch.randn(1000, data_dim)  # 生成一些随机数据
dataset = TensorDataset(x)
dataloader = DataLoader(dataset, batch_size=32)

# 训练模型
model = GAN(noise_dim, data_dim)
trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1)
trainer.fit(model, dataloader)
