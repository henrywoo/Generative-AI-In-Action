"""
🌳 ViTForImageClassification<all params:86,567,656>
├── ViTModel(vit)
│   ├── ViTEmbeddings(embeddings)|cls_token[1,1,768]|position_embeddings[1,197,768]
│   │   └── ViTPatchEmbeddings(patch_embeddings)
│   │       └── Conv2d(projection)|weight[768,3,16,16]🇸 -(16, 16)|bias[768]🇸-(16, 16)
│   ├── ViTEncoder(encoder)
│   │   └── ModuleList(layer)
│   │       └── 💠 ViTLayer(0-11)<🦜:7087872x12>
│   │           ┣━━ ViTAttention(attention)
│   │           ┃   ┣━━ ViTSelfAttention(attention)
│   │           ┃   ┃   ┗━━ 💠 Linear(query,key,value)<🦜:590592x3>|weight[768,768]|bias[768]
│   │           ┃   ┗━━ ViTSelfOutput(output)
│   │           ┃       ┗━━ Linear(dense)|weight[768,768]|bias[768]
│   │           ┣━━ ViTIntermediate(intermediate)
│   │           ┃   ┗━━ Linear(dense)|weight[3072,768]|bias[3072]
│   │           ┣━━ ViTOutput(output)
│   │           ┃   ┗━━ Linear(dense)|weight[768,3072]|bias[768]
│   │           ┗━━ 💠 LayerNorm(layernorm_before,layernorm_after)<🦜:1536x2>|weight[768]|bias[768]
│   └── LayerNorm(layernorm)|weight[768]|bias[768]
└── Linear(classifier)|weight[1000,768]|bias[1000]
Predicted class: Egyptian cat
"""
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from hiq.vis import print_model

# Load model and feature extractor
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
print_model(model)
# Load and preprocess image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted class
predicted_class_id = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_id])