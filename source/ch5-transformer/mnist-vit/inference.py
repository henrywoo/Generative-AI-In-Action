from dataset import MNIST
import matplotlib.pyplot as plt
import torch
from vit import ViT
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

dataset = MNIST()  # 数据集

model = ViT().to(DEVICE)  # 模型
model.load_state_dict(torch.load('model.pth'))

model.eval()  # 预测模式

image, label = dataset[0]
print('Correct classification:', label)
plt.imshow(image.permute(1, 2, 0))
plt.show()

with torch.no_grad():
    logits = model(image.unsqueeze(0).to(DEVICE))
    print('Predicted classification:', logits.argmax(-1).item())
