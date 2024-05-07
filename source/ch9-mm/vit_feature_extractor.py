"""
The google/vit-huge-patch14-224-in21k ViT model is pretrained on the ImageNet-21k dataset, which has 21,841 classes.
This model primarily outputs image features (the last_hidden_states), not direct class predictions.
To get class names, you'd need a classification head.
🌳 ViTModel<all params:632,404,480>
├── ViTEmbeddings(embeddings)|cls_token[1,1,1280]|position_embeddings[1,257,1280
│   ]
│   └── ViTPatchEmbeddings(patch_embeddings)
│       └── Conv2d(projection)|weight[1280,3,14,14]🇸 -(14, 14)|bias[1280]🇸 -(14,
│           14)
├── ViTEncoder(encoder)
│   └── ModuleList(layer)
│       └── 💠 ViTLayer(0-31)<🦜:19677440x32>
│           ┣━━ ViTAttention(attention)
│           ┃   ┣━━ ViTSelfAttention(attention)
│           ┃   ┃   ┗━━ 💠
│           ┃   ┃       Linear(query,key,value)<🦜:1639680x3>|weight[1280,1280]|
│           ┃   ┃       bias[1280]
│           ┃   ┗━━ ViTSelfOutput(output)
│           ┃       ┗━━ Linear(dense)|weight[1280,1280]|bias[1280]
│           ┣━━ ViTIntermediate(intermediate)
│           ┃   ┗━━ Linear(dense)|weight[5120,1280]|bias[5120]
│           ┣━━ ViTOutput(output)
│           ┃   ┗━━ Linear(dense)|weight[1280,5120]|bias[1280]
│           ┗━━ 💠
│               LayerNorm(layernorm_before,layernorm_after)<🦜:2560x2>|weight[12
│               80]|bias[1280]
├── LayerNorm(layernorm)|weight[1280]|bias[1280]
└── ViTPooler(pooler)
    └── Linear(dense)|weight[1280,1280]|bias[1280]
"""

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
from hiq.vis import print_model

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-huge-patch14-224-in21k')
model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')
print_model(model)

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)