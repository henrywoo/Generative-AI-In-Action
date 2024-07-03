import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from skimage import data
from torch.nn.modules.utils import _pair
from torch import nn
from hiq import deterministic

class CausalConv2d(nn.Conv2d):
    """https://gist.github.com/wassname/7eb4095a4f3d3b5eea8adaaf4419c822"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding is None:
            padding = [int((kernel_size[i] -1) * dilation[i]) for i in range(len(kernel_size))]
        else:
           padding = padding * 2
        self.left_padding = _pair(padding)
        super().__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)

    def forward(self, inputs):
        inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super().forward(inputs)
        return output

# Load the astronaut image
image = data.astronaut()
image_tensor = ToTensor()(image).unsqueeze(0)

# Apply CausalConv2d
conv = CausalConv2d(3, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
output_tensor = conv(image_tensor)

# Convert tensors back to images
output_image = ToPILImage()(output_tensor.squeeze(0))

# Plot original and processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Causal Conv2d Image')
plt.axis('off')

plt.tight_layout()
plt.show()
