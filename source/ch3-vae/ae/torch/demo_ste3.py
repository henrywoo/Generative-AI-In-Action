import matplotlib.pyplot as plt
import numpy as np
import torch

def soft_round(x, alpha=10.0):
    """Continuous and differentiable approximation to the round function."""
    return torch.round(x) + torch.tanh(alpha * (x - torch.round(x))) / torch.tanh(torch.tensor(alpha))

# Generate data for plotting
x = torch.linspace(-2, 2, steps=400)
y = soft_round(x)

# Convert tensors to numpy arrays for plotting
x_np = x.numpy()
y_np = y.detach().numpy()

# Plotting the soft_round function
plt.figure(figsize=(10, 6))
plt.plot(x_np, y_np, label='soft_round(x)', color='blue')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.axvline(x=0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title('Soft Round Function')
plt.xlabel('x')
plt.ylabel('soft_round(x)')
plt.legend()
plt.show()
