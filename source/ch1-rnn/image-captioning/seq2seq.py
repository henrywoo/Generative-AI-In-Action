#!/usr/bin/env python

import torch
import torchvision
import random
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from hiq.vis import print_model

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_data():
    """Load MNIST data."""
    train_data = torchvision.datasets.MNIST('files/', train=True, download=True)
    test_data = torchvision.datasets.MNIST('files/', train=False, download=True)
    return train_data, test_data

class MyDataSet(Dataset):
    def __init__(self, data, label, batch_size):
        self.index = 0
        self.total = data.shape[0]
        self.batch = batch_size
        self.data = data.to(torch.float) / 255
        self.label = label
        self.point = torch.zeros((28, 28), dtype=torch.float)
        self.point[20:23, 12:15] = 1

    def __getitem__(self, idx):
        dot = np.random.randint(1, 6)
        y = random.Random(int(idx // self.batch)).randint(4, 8)
        a = torch.zeros((int(28 * 2), 28 * y), dtype=torch.float)
        target = [[10]]

        for x in range(y):
            yy = np.random.randint(0, 28)
            g = np.random.randint(1, 5)
            if x != dot:
                a[yy:yy + 28, x * 28 + g:(28 * x) + 28 - g] = self.data[self.index][:, g:-g]
                target.append([self.label[self.index]])
            else:
                a[yy:yy + 28, x * 28 + g:(28 * x) + 28 - g] = self.point[:, g:-g]
                target.append([12])
            self.index = (self.index + 1) % self.total
        target.append([11])
        return a.rot90(3).unsqueeze(0), torch.tensor(target[:-1]), torch.tensor(target[1:])

    def __len__(self):
        return self.total

class EncoderDecoder(nn.Module):
    def __init__(self, output):
        super(EncoderDecoder, self).__init__()
        self.output = output
        self.maxpool1 = nn.MaxPool2d(2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.cnn1 = nn.Conv2d(1, 8, 3)
        self.cnn2 = nn.Conv2d(8, 8, 3)
        self.cnn3 = nn.Conv2d(8, 4, 3)
        self.encgru = nn.GRU(40, 32, 2, batch_first=True, dropout=0.1)
        self.emb = nn.Embedding(self.output, 8)
        self.decgru = nn.GRU(8, 32, 2, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(32, self.output)

    def forward(self, x, val):
        x = x.to(device)
        val = val.to(device)
        x = self.cnn1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = nn.functional.relu(x)
        batch, channel, time, emb = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, emb * channel)
        _, hidden = self.encgru(x)
        x = self.emb(val)
        x = nn.functional.relu(x)
        x = x.squeeze(2)
        x, _ = self.decgru(x, hidden)
        x = nn.functional.relu(x)
        x = self.linear(x.reshape(-1, 32))
        return x

    def predict(self, x):
        x = x.to(device)
        x = self.cnn1(x)
        x = nn.functional.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = nn.functional.relu(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = nn.functional.relu(x)
        batch, channel, time, emb = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, emb * channel)
        _, hidden = self.encgru(x)
        index = 10
        pred = [index]
        for _ in range(12):
            x = self.emb(torch.tensor([[[index]]], device=device))
            x = nn.functional.relu(x)
            x = x.squeeze(2)
            x, hidden = self.decgru(x, hidden)
            x = nn.functional.relu(x)
            x = self.linear(x.reshape(-1, 32))
            index = torch.argmax(x, -1)[0]
            pred.append(index.item())
            if index == 11:
                break
        return pred

def train_model(model, train_loader, test_loader, device, num_epochs=40, save_path='best_ocr.pt'):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    best_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for x in train_loader:
            out = model(x[0].to(device), x[1].to(device))
            optimizer.zero_grad()
            loss = loss_fn(out, x[2].view(-1).to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        val_acc = validate_model(model, test_loader, device)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_acc:.4f}')

    model.load_state_dict(torch.load(save_path))

    plot_curves(train_losses, val_accuracies)

    return model

def validate_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for x in test_loader:
            out = model(x[0].to(device), x[1].to(device))
            preds = torch.argmax(out, -1).cpu()
            total_correct += (preds == x[2].view(1, -1)).sum().item()
            total_count += preds.shape[-1]
    model.train()
    return total_correct / total_count

def plot_curves(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    with plt.style.context('ggplot'):
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'r', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_accuracies, 'b', label='Validation accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('Training_loss_validation_accuracy.png')
        plt.show()


def plot_predictions(model, test_loader, save_path='predictions.jpg'):
    model.eval()
    map_index = {12: "."}

    _, axs = plt.subplots(6, 3, figsize=(18, 18))
    axs = axs.flatten()
    for i, (x, ax) in enumerate(zip(test_loader, axs)):
        if i == 18:
            break
        ax.imshow(x[0][0, 0].cpu().numpy().transpose()[::-1])
        out = model.predict(x[0])
        filtered_out = [i for i in out if i not in [10, 11]]  # Filter out <START> (10) and <END> (11)
        ax.set_title("".join([map_index.get(i, str(i)) for i in filtered_out]), color="black", fontsize=18)
    plt.savefig(save_path)

def main():
    train_data, test_data = load_data()
    train_dataset = MyDataSet(train_data.data, train_data.targets, batch_size=8)
    test_dataset = MyDataSet(test_data.data, test_data.targets, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = EncoderDecoder(output=13).to(device)
    print_model(model)
    trained_model = train_model(model, train_loader, test_loader, device, num_epochs=30)

    plot_predictions(trained_model, test_loader)

if __name__ == "__main__":
    main()
