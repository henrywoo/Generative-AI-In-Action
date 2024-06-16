import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import matplotlib.pyplot as plt
import numpy as np


def load_model_and_feature_extractor(model_name):
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, output_attentions=True)
    model = ViTModel.from_pretrained(model_name, output_attentions=True)
    return feature_extractor, model


def preprocess_image(image_path, feature_extractor):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return image, inputs


def get_attention_map(attentions, layer_index, image_size, head_index):
    attention = attentions[layer_index]  # Get the attention weights of the chosen layer
    if head_index is None:
        attention = attention.mean(dim=1)  # Take the mean of attention weights over all heads
        attention = attention[0, :, 1:]  # Exclude the class token
    else:
        # Take the attention head at index head_index and exclude the class token for both axes
        attention = attention[0, head_index, 1:, 1:]
    attention = attention.detach().cpu().numpy()
    h_featmap = w_featmap = int(np.sqrt(attention.shape[-1]))
    attention = attention.reshape(-1, h_featmap, w_featmap)
    attention = torch.nn.functional.interpolate(
        torch.tensor(attention).unsqueeze(0),
        size=image_size[::-1],
        mode="nearest"
    )[0].cpu().numpy()
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    return attention

def plot_attention_maps(image, attentions, save_path, head_index):
    num_layers = len(attentions)
    layers_to_display = [0, num_layers // 3, 2 * num_layers // 3, num_layers - 1]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    plt.style.use('ggplot')
    plt.suptitle(f"Attention Map At Head #{head_index}")

    for ax, layer_idx in zip(axes.flat, layers_to_display):
        attention = get_attention_map(attentions, layer_idx, image.size, head_index)
        ax.imshow(image)
        ax.imshow(attention[0], cmap='jet', alpha=0.5, interpolation='nearest')
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path + f"_{head_index}.png")
    plt.show()


# Load the model and feature extractor
model_name = "facebook/dino-vits8"
feature_extractor, model = load_model_and_feature_extractor(model_name)

# Load and preprocess the image
image_path = 'img/cats.jpg'
image, inputs = preprocess_image(image_path, feature_extractor)

# Get model outputs including attention weights
outputs = model(**inputs)
# [batch_size, num_heads, sequence_length, sequence_length]
attentions = outputs.attentions

# Plot the attention maps
for h in range(6):
    plot_attention_maps(image, attentions, 'img/attention_map_dino', head_index=h)
