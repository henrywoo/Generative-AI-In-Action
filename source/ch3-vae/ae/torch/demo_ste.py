import torch
import torch.nn as nn
import torch.optim as optim
from hiq import set_seed

set_seed(0xdeadbeef, has_torch=True)
def round_naive(x: torch.Tensor) -> torch.Tensor:
    zhat = x.round()
    return zhat

def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = x.round()
    y = x + (zhat - x).detach()
    return y


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        t0 = self.linear(x)
        t1 = round_ste(t0)
        return t1

model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

x = torch.tensor([[1.0]], requires_grad=True)
y = torch.tensor([[1.0]])
print("Initial weights:", model.linear.weight.data)
print("Initial bias:", model.linear.bias.data)

output = model(x)
loss = criterion(output, y)
print(f"Initial loss:{loss.item():.2f}")
optimizer.zero_grad()
loss.backward()
print("Gradients on weights before step:", model.linear.weight.grad)
print("Gradients on bias before step:", model.linear.bias.grad)
optimizer.step()
print("Updated weights:", model.linear.weight.data)
print("Updated bias:", model.linear.bias.data)

output = model(x)
loss = criterion(output, y)
print("Updated loss:", loss.item())
print("Final weights:", model.linear.weight.data)
print("Final bias:", model.linear.bias.data)
