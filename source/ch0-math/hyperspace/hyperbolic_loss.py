import torch
import torch.nn as nn
from hiq import deterministic


# 安全计算acosh的方法，避免数值不稳定性
def safe_acosh(x, eps=1e-12):
    return torch.log(x + torch.sqrt(x ** 2 - 1 + eps))


class PoincareDistance(nn.Module):
    def __init__(self):
        super(PoincareDistance, self).__init__()

    def forward(self, u, v):
        norm_u = torch.clamp(torch.norm(u, dim=-1, keepdim=True), max=0.999)
        norm_v = torch.clamp(torch.norm(v, dim=-1, keepdim=True), max=0.999)
        diff = u - v
        norm_diff = torch.norm(diff, dim=-1, keepdim=True)

        num = 2 * norm_diff ** 2
        denom = (1 - norm_u ** 2) * (1 - norm_v ** 2)

        return safe_acosh(1 + num / denom)


class HyperbolicLoss(nn.Module):
    def __init__(self):
        super(HyperbolicLoss, self).__init__()
        self.poincare_distance = PoincareDistance()

    def forward(self, predicted_points, true_points):
        distances = self.poincare_distance(predicted_points, true_points)
        return torch.mean(distances ** 2)


# 示例数据
predicted_points = torch.rand(10, 5) * 0.9  # 预测的高维点（例如5维）
true_points = torch.rand(10, 5) * 0.9  # 真实的高维点

# 创建损失函数
loss_fn = HyperbolicLoss()

# 计算损失
loss = loss_fn(predicted_points, true_points)
print(f'Hyperbolic Loss: {loss.item()}')
