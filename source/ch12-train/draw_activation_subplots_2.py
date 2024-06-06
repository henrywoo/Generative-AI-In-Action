# https://aew61.github.io/blog/artificial_neural_networks/1_background/1.b_activation_functions_and_derivatives.html
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
    dy = sig + x * sig * (1 - sig)
    return y, dy


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
    (sigmoid, 'Sigmoid'),
    (tanh, 'Tanh'),
    (relu, 'ReLU'),
    (leaky_relu, 'Leaky-ReLU'),
    (elu, 'ELU'),
    (selu, 'SELU'),
    (gelu, 'GELU'),
    (swish, 'Swish'),
    (mish, 'Mish')
]
# Prepare the figure and subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()

for i, (func, name) in enumerate(functions):
    y, dy = func(x)

    # Fill areas above and below y=0
    axes[i].axhspan(0, max(max(y), max(dy)), facecolor='lightgreen', alpha=0.3)
    axes[i].axhspan(min(min(y), min(dy)), 0, facecolor='yellow', alpha=0.3)

    axes[i].plot(x, y, label=f'{name}')
    axes[i].plot(x, dy, label=f'{name} Grad', linestyle='--')
    axes[i].set_title(f'{name}', fontsize=8)
    axes[i].set_xlabel('Input(x)', fontsize=8)
    axes[i].set_ylabel('Output / Gradient', fontsize=8)
    axes[i].legend(fontsize=8)
    axes[i].grid(True)

# Remove unused subplots
for j in range(len(functions), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("activation_functions_and_gradients_2.png")
plt.show()
