import torch

# 原始张量
z = torch.tensor([1.2, 2.7, 3.3], requires_grad=True)

# 使用 STE 进行舍入
zhat_ste = z.round()
z_ste = z + (zhat_ste - z).detach()

# 计算损失并进行反向传播
loss_ste = z_ste.sum()
loss_ste.backward()
print("Using STE - Gradients:", z.grad)

# 重置梯度
z.grad.zero_()

# 不使用 STE 直接舍入
z_rounded = z.round()

# 计算损失并进行反向传播
loss_rounded = z_rounded.sum()
loss_rounded.backward()
print("Without STE - Gradients:", z.grad)
