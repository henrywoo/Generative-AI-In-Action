import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x), (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x), alpha + (x > 0) * (1 - alpha)


def elu(x, alpha=1.0):
    y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    dy = np.where(x > 0, 1, alpha * np.exp(x))
    return y, dy


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    y = scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    dy = scale * np.where(x > 0, 1, alpha * np.exp(x))
    return y, dy


def gelu(x):
    c = np.sqrt(2 / np.pi)
    y = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x ** 3)))
    dx = 0.5 * (1 + np.tanh(c * (x + 0.044715 * x ** 3))) + \
         0.5 * x * c * (1 - np.tanh(c * (x + 0.044715 * x ** 3)) ** 2) * \
         (1 + 3 * 0.044715 * x ** 2)
    return y, dx


def swish(x):
    y = x * (1 / (1 + np.exp(-x)))  # Or use x * sigmoid(x)
    sig = 1 / (1 + np.exp(-x))
    dx = sig + x * sig * (1 - sig)
    return y, dx


def mish(x):
    softplus = np.log(1 + np.exp(x))
    y = x * np.tanh(softplus)
    dy = np.tanh(softplus) + x * (1 - np.tanh(softplus) ** 2) * (1 / (1 + np.exp(-x)))
    return y, dy


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    dy = y * (1 - y)
    return y, dy


def tanh(x):
    y = np.tanh(x)
    dy = 1 - y ** 2
    return y, dy


# Create input data
x = np.linspace(-10, 10, 200)

plt.style.use('ggplot')
# Activation functions and their gradients
functions = [
    (relu, 'ReLU'),
    (leaky_relu, 'LReLU'),
    (elu, 'ELU'),
    (selu, 'SELU'),
    (gelu, 'GELU'),
    (swish, 'Swish'),
    (mish, 'Mish'),
    (sigmoid, 'Sigmoid'),
    (tanh, 'Tanh')
]
# Prepare the figure and subplots
fig, axes = plt.subplots(3, 6, figsize=(16, 7))
axes = axes.flatten()

for i, (func, name) in enumerate(functions):
    y, dy = func(x)

    # Fill areas above and below y=0
    axes[2 * i].axhspan(0, max(y), facecolor='lightgreen', alpha=0.3)
    axes[2 * i].axhspan(min(y), 0, facecolor='yellow', alpha=0.3)
    axes[2 * i + 1].axhspan(0, max(dy), facecolor='lightgreen', alpha=0.3)
    axes[2 * i + 1].axhspan(min(dy), 0, facecolor='yellow', alpha=0.3)

    axes[2 * i].plot(x, y, label=f'{name}')
    axes[2 * i].set_title(f'{name}', fontsize=8)
    axes[2 * i].set_ylabel('Output', fontsize=8)
    axes[2 * i].legend(fontsize=8)
    axes[2 * i].grid(True)

    axes[2 * i + 1].plot(x, dy, label=f'{name} Grad', linestyle='--')
    axes[2 * i + 1].set_title(f'{name} Grad', fontsize=8)
    axes[2 * i + 1].set_ylabel('Gradient', fontsize=8)
    axes[2 * i + 1].legend(fontsize=8)
    axes[2 * i + 1].grid(True)

# Remove unused subplots
for j in range(2 * len(functions), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("activation_functions_and_gradients.png")
plt.show()
