from config import *
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)

# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x)

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
autoencoder = Autoencoder()

# Define the optimizer
optimizer = optim.SGD(autoencoder.parameters(), lr=0.5)
criterion = nn.MSELoss()

# Generate the dataset
m = 60
X = np.zeros((m, 3))  # initialize 3D dataset
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # oval
X += 0.28 * np.random.randn(m, 3)  # add more noise
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X_train = X + [0.2, 0, 0.2]  # shift a bit

X_train = torch.tensor(X_train, dtype=torch.float32)

# Train the autoencoder
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(X_train)
    loss = criterion(outputs, X_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Get the codings (encoded representations)
with torch.no_grad():
    codings = autoencoder.encoder(X_train).numpy()

# Plot the codings
fig = plt.figure(figsize=(4, 3))
plt.plot(codings[:, 0], codings[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
plt.title("Linear Autoencoder PCA Plot")
save_fig("linear_autoencoder_pca_plot")
plt.show()

# Plot X_train in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c='b', marker='o')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title("3D Plot of X_train")
save_fig("3d_plot_x_train")
plt.show()
