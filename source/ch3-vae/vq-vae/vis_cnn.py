from hiq import print_model
import torchvision.models as models
import matplotlib.pyplot as plt

# 使用预训练的ResNet模型
model = models.resnet18(pretrained=True)
print_model(model)

# 提取第一个卷积层的权重
conv1_weights = model.conv1.weight.data

# 可视化卷积核
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < conv1_weights.size(0):
        ax.imshow(conv1_weights[i].permute(1, 2, 0).numpy())
    ax.axis('off')

plt.show()
