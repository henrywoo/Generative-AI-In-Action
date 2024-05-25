import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from hiq.vis import print_model

"""
🌳 VGG<all params:138357544>
├── Sequential(features)
│   ├── Conv2d(0)|weight[64,3,3,3]|bias[64]
│   ├── Conv2d(2)|weight[64,64,3,3]|bias[64]
│   ├── Conv2d(5)|weight[128,64,3,3]|bias[128]
│   ├── Conv2d(7)|weight[128,128,3,3]|bias[128]
│   ├── Conv2d(10)|weight[256,128,3,3]|bias[256]
│   ├── 💠 Conv2d(12-12,14-14)<🦜:590080x2>|weight[256,256,3,3]|bias[256]
│   ├── Conv2d(17)|weight[512,256,3,3]|bias[512]
│   └── 💠 Conv2d(19-19,21-21,24-24,26-26,28-28)<🦜:2359808x5>|weight[512,512,3,3]|bias[512]
└── Sequential(classifier)
    ├── Linear(0)|weight[4096,25088]|bias[4096]
    ├── Linear(3)|weight[4096,4096]|bias[4096]
    └── Linear(6)|weight[1000,4096]|bias[1000]
"""
def load_vgg16_model():
    model = models.vgg16(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_tensor

def get_feature_map(model, layer_idx, input_tensor):
    # Pass the image through the model up to the chosen layer
    layer_output = None
    def hook_fn(module, input, output):
        nonlocal layer_output
        layer_output = output

    handle = model.features[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_tensor)
    handle.remove()

    return layer_output.squeeze(0).cpu().numpy()  # Remove the batch dimension

CONV_LAYER_INDICES = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]  # Conv2d layer indices for VGG16
def plot_feature_maps(feature_maps, save_path):
    num_layers = len(feature_maps)
    layers_to_display = [0, num_layers // 3, 2 * num_layers // 3, num_layers - 1]
    for i, k in enumerate([0, 8, 16, 24, 32, 40]):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        plt.style.use('ggplot')
        plt.suptitle(f"VGG16 Feature Maps @ index: {k}")
        for ax, layer_idx in zip(axes.flat, layers_to_display):
            feature_map = feature_maps[layer_idx][k]
            ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Layer {CONV_LAYER_INDICES[layer_idx]}', fontsize=8)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path + f"_{i}.png")
        plt.show()

# Load the VGG16 model
model = load_vgg16_model()
print_model(model)

# Load and preprocess the image
image_path = 'img/cats.jpg'
input_tensor = preprocess_image(image_path)

# Get feature maps from selected layers
feature_maps = [get_feature_map(model, idx, input_tensor) for idx in CONV_LAYER_INDICES]

# Plot the feature maps
plot_feature_maps(feature_maps, 'img/feature_map_vgg16')
