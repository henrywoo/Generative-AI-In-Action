# Transformer

## VIT (Vision Transformer)

![](vision-transformer-vit.png)

```
🌳 ViT<all params:207786>
├── Conv2d(conv)|weight[16,1,4,4]🇸 -(4, 4)|bias[16]🇸 -(4, 4)
├── Linear(patch_emb)|weight[16,16]|bias[16]
├── TransformerEncoder(tranformer_enc)
│   └── ModuleList(layers)
│       └── 💠 TransformerEncoderLayer(0-2)<🦜:68752x3>
│           ┣━━ MultiheadAttention(self_attn)|in_proj_weight[48,16]|in_proj_bias[48]
│           ┃   ┗━━ NonDynamicallyQuantizableLinear(out_proj)|weight[16,16]|bias[16]
│           ┣━━ Linear(linear1)|weight[2048,16]|bias[2048]
│           ┣━━ Linear(linear2)|weight[16,2048]|bias[16]
│           ┗━━ 💠 LayerNorm(norm1,norm2)<🦜:32x2>|weight[16]|bias[16]
└── Linear(cls_linear)|weight[10,16]|bias[10]
```

![](img/attention_map_dino_0.png)

In the resulting image, the red areas in the attention map over the cat's head indicate that these patches are receiving a high amount of attention. This means that the model is focusing heavily on these areas when processing the image. Specifically:

1. **High Attention Values**: The red areas correspond to high attention weights. This implies that the model considers the information from these patches (the cat's head) to be very important.

2. **Feature Importance**: The model is likely identifying the cat's head as a critical feature for understanding or classifying the image. In many tasks, the head of an animal can be a distinguishing feature due to the presence of eyes, nose, and other significant details.

3. **Contextual Relevance**: High attention on the cat's head suggests that these patches provide crucial contextual information. The model might be using these features to infer more about the rest of the image or to make sense of the scene.

### Summary of Attention Map Interpretation:
- **Red Areas**: High attention, indicating important features or context for the model.
- **Green/Yellow Areas**: Moderate attention, still relevant but less critical than red areas.
- **Blue Areas**: Low attention, indicating less relevance to the current focus of the model.

In this specific case, the model focuses heavily on the cat's head in the later layers (Layer 12), suggesting that it has learned to recognize the importance of this region for the image's overall interpretation. This is a common behavior in vision models, where certain key features (like faces or heads) are given more importance during the attention process.
