import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    # Pre-calculated values for SELU
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    return x * (1 / (1 + np.exp(-x)))  # Or use x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# Create input data
x = np.linspace(-5, 5, 200)

# Plot all functions
plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.plot(x, elu(x), label='ELU')
plt.plot(x, selu(x), label='SELU')
plt.plot(x, gelu(x), label='GELU')
plt.plot(x, swish(x), label='Swish')
plt.plot(x, mish(x), label='Mish')
plt.plot(x, sigmoid(x), label='Sigmoid')  # Added
plt.plot(x, tanh(x), label='Tanh')        # Added

# Grid and style
plt.grid(True)


# Labels and legend
plt.xlabel('Input (x)')
plt.ylabel('Output')
plt.title('Comparison of Activation Functions')
plt.legend()

plt.tight_layout()
plt.savefig("activations.png")
plt.show()

