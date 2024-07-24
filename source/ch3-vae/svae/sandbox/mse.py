import torch
import torch.nn.functional as F

# Hardcoded tensors x and y
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

y = torch.tensor([[1.5, 2.5, 3.5],
                  [4.5, 5.5, 6.5]])

# Compute MSE loss with reduction='sum'
mse_loss_sum = F.mse_loss(x, y, reduction='sum')

# Compute MSE loss with reduction='mean'
mse_loss_mean = F.mse_loss(x, y, reduction='mean')

# Print the results
print(f"Input tensor x:\n{x}\n")
print(f"Target tensor y:\n{y}\n")
print(x.numel())
m = mse_loss_sum.item()
n = mse_loss_mean.item()

print(f"MSE loss with reduction='sum': {m}")
print(f"MSE loss with reduction='mean': {n}")
print(f"m/x.numel()={m/x.numel()}")