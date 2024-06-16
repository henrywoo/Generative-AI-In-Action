from transformers import AutoImageProcessor, AutoModel
import requests
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from hiq import print_model

"""
🌳 Dinov2Model<all params:86580480>
├── Dinov2Embeddings(embeddings)|cls_token[1,1,768]|mask_token[1,768]|position_embeddings[1,1370,768]
│   └── Dinov2PatchEmbeddings(patch_embeddings)
│       └── Conv2d(projection)|weight[768,3,14,14]🇸 -(14, 14)|bias[768]🇸 -(14, 14)
├── Dinov2Encoder(encoder)
│   └── ModuleList(layer)
│       └── 💠 Dinov2Layer(0-11)<🦜:7089408x12>
│           ┣━━ 💠 LayerNorm(norm1,norm2)<🦜:1536x2>|weight[768]|bias[768]
│           ┣━━ Dinov2Attention(attention)
│           ┃   ┣━━ Dinov2SelfAttention(attention)
│           ┃   ┃   ┗━━ 💠 Linear(query,key,value)<🦜:590592x3>|weight[768,768]|bias[768]
│           ┃   ┗━━ Dinov2SelfOutput(output)
│           ┃       ┗━━ Linear(dense)|weight[768,768]|bias[768]
│           ┣━━ 💠 Dinov2LayerScale(layer_scale1,layer_scale2)<🦜:768x2>|lambda1[768]
│           ┗━━ Dinov2MLP(mlp)
│               ┣━━ Linear(fc1)|weight[3072,768]|bias[3072]
│               ┗━━ Linear(fc2)|weight[768,3072]|bias[768]
└── LayerNorm(layernorm)|weight[768]|bias[768]
"""

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base', output_attentions=True, output_hidden_states=True)
print_model(model)
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
attentions = outputs.attentions

def process_attentions(attention, image_size):
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

def get_attention_map(attentions, layer_index, head_index, image_size):
    attention = attentions[layer_index]  # Get the attention weights of the chosen layer
    if head_index is None:
        attention = attention.mean(dim=1)  # Take the mean of attention weights over all heads
        attention = attention[0, :, 1:]  # Exclude the class token
    else:
        # Take the attention head at index head_index and exclude the class token for both axes
        attention = attention[0, head_index, 1:, 1:]
    return process_attentions(attention, image_size)

def plot_attention_maps(image, attentions, save_path, head_index):
    num_layers = len(attentions)
    layers_to_display = [0, num_layers // 3, 2 * num_layers // 3, num_layers - 1]
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    plt.style.use('ggplot')
    plt.suptitle(f"Attention Map At Head #{head_index}")

    for ax, layer_idx in zip(axes.flat, layers_to_display):
        attention = get_attention_map(attentions, layer_idx, head_index, image.size)
        ax.imshow(image)
        ax.imshow(attention[0], cmap='jet', alpha=0.5, interpolation='nearest')
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path + f"_{head_index}.png")
    plt.show()

# Plot the attention maps
for h in range(1):
    plot_attention_maps(image, attentions, 'img/attention_map_dino', head_index=h)