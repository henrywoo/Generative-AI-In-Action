import numpy as np
import matplotlib.pyplot as plt

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
    y = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))
    dx = 0.5 * (1 + np.tanh(c * (x + 0.044715 * x**3))) + \
         0.5 * x * c * (1 - np.tanh(c * (x + 0.044715 * x**3))**2) * \
         (1 + 3 * 0.044715 * x**2)
    return y, dx

def swish(x):
    y = x * (1 / (1 + np.exp(-x)))  # Or use x * sigmoid(x)
    sig = 1 / (1 + np.exp(-x))
    dx = sig + x * sig * (1 - sig)
    return y, dx

def mish(x):
    softplus = np.log(1 + np.exp(x))
    y = x * np.tanh(softplus)
    dy = np.tanh(softplus) + x * (1 - np.tanh(softplus)**2) * (1 / (1 + np.exp(-x)))
    return y, dy

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    dy = y * (1 - y)
    return y, dy

def tanh(x):
    y = np.tanh(x)
    dy = 1 - y**2
    return y, dy

# Create input data
x = np.linspace(-5, 5, 200)


plt.style.use('ggplot')

# Prepare the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot all functions
y_leaky_relu, dy_leaky_relu = leaky_relu(x)
y_elu, dy_elu = elu(x)
y_selu, dy_selu = selu(x)
y_gelu, dy_gelu = gelu(x)
y_swish, dy_swish = swish(x)
y_mish, dy_mish = mish(x)
y_sigmoid, dy_sigmoid = sigmoid(x)
y_tanh, dy_tanh = tanh(x)

ax1.plot(x, y_leaky_relu, label='Leaky ReLU')
ax1.plot(x, y_elu, label='ELU')
ax1.plot(x, y_selu, label='SELU')
ax1.plot(x, y_gelu, label='GELU')
ax1.plot(x, y_swish, label='Swish')
ax1.plot(x, y_mish, label='Mish')
ax1.plot(x, y_sigmoid, linestyle=':', label='Sigmoid (dotted)')
ax1.plot(x, y_tanh, linestyle=':', label='Tanh (dotted)')
ax1.set_title('Activation Functions')
ax1.set_xlabel('Input (x)')
ax1.set_ylabel('Output')
ax1.legend()

# Plot all gradients
ax2.plot(x, dy_leaky_relu, label='Leaky ReLU')
ax2.plot(x, dy_elu, label='ELU')
ax2.plot(x, dy_selu, label='SELU')
ax2.plot(x, dy_gelu, label='GELU')
ax2.plot(x, dy_swish, label='Swish')
ax2.plot(x, dy_mish, label='Mish')
ax2.plot(x, dy_sigmoid, linestyle=':', label='Sigmoid (dotted)')
ax2.plot(x, dy_tanh, linestyle=':', label='Tanh (dotted)')
ax2.set_title('Gradients of Activation Functions')
ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Gradient')
ax2.legend()

plt.tight_layout()
plt.savefig("activation_functions_and_gradients.png")
plt.show()
