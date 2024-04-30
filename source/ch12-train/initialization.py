import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Parameters
input_dim = 100
output_dim = 5

# Initialize models with different methods
model_glorot_uniform = LinearModel(input_dim, output_dim)
torch.nn.init.xavier_uniform_(model_glorot_uniform.linear.weight)

model_glorot_normal = LinearModel(input_dim, output_dim)
torch.nn.init.xavier_normal_(model_glorot_normal.linear.weight)

model_lecun_uniform = LinearModel(input_dim, output_dim)
torch.nn.init.normal_(model_lecun_uniform.linear.weight, std=1 / (input_dim ** 0.5))  # LeCun Uniform

model_kaiming_uniform = LinearModel(input_dim, output_dim)
torch.nn.init.kaiming_uniform_(model_kaiming_uniform.linear.weight, nonlinearity='relu')  # He Uniform

# Collect weight distributions
weights_glorot_uniform = model_glorot_uniform.linear.weight.detach().flatten().numpy()
weights_glorot_normal = model_glorot_normal.linear.weight.detach().flatten().numpy()
weights_lecun_uniform = model_lecun_uniform.linear.weight.detach().flatten().numpy()
weights_kaiming_uniform = model_kaiming_uniform.linear.weight.detach().flatten().numpy()

# Plot the distributions as histograms
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.hist(weights_glorot_uniform, bins=25, label='Glorot Uniform')
plt.title('Glorot Uniform Distribution')
plt.legend()

plt.subplot(2, 2, 2)
plt.hist(weights_glorot_normal, bins=25, label='Glorot Normal')
plt.title('Glorot Normal Distribution')
plt.legend()

plt.subplot(2, 2, 3)
plt.hist(weights_lecun_uniform, bins=25, label='LeCun Uniform')
plt.title('LeCun Uniform Distribution')
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(weights_kaiming_uniform, bins=25, label='Kaiming Uniform')
plt.title('Kaiming Uniform Distribution')
plt.legend()

plt.tight_layout()
plt.savefig("initialization.png")
plt.show()
