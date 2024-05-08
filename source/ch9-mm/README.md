# Multimodal AI

## VIT

```angular2html
🌳 ViT<all params:207786>
├── Conv2d(conv)|weight[16,1,4,4]🇸 -(4, 4)|bias[16]🇸 -(4, 4)
├── Linear(patch_emb)|weight[16,16]|bias[16]
├── TransformerEncoder(tranformer_enc)
│   └── ModuleList(layers)
│       └── 💠 TransformerEncoderLayer(0-2)<🦜:68752x3>
│           ┣━━ MultiheadAttention(self_attn)|in_proj_weight[48,16]|in_proj_bias
│           ┃   [48]
│           ┃   ┗━━ NonDynamicallyQuantizableLinear(out_proj)|weight[16,16]|bias
│           ┃       [16]
│           ┣━━ Linear(linear1)|weight[2048,16]|bias[2048]
│           ┣━━ Linear(linear2)|weight[16,2048]|bias[16]
│           ┗━━ 💠 LayerNorm(norm1,norm2)<🦜:32x2>|weight[16]|bias[16]
└── Linear(cls_linear)|weight[10,16]|bias[10]
```

## SAM

![](https://learnopencv.com/wp-content/uploads/2023/04/segment-anything-model.png)

### In segment anything model, how the text embeddings and image embeddings are fused?

In the Segment Anything Model (SAM), text embeddings and image embeddings are fused in a few key ways:

**1. Prompt Encoder**

* **Sparse Prompts (Points, Boxes, Text):**
    * Positional embedding for spatial information (location of points or boxes).
    * Learned embeddings represent different prompt types (point, box, text) 
    * Text is directly encoded using a text encoder (e.g., CLIP)

* **Dense Prompts (Masks):**
    * Downsampled to match image resolution.
    * Convolved and embedded with learned weights.

**2. Fusion via Element-Wise Addition**

* **The Core Mechanism:** The central way SAM fuses image and prompt embeddings is by simple element-wise addition. After processing, both prompt embeddings and image embeddings have the same dimensionality (256 channels).

* **Handling No Mask:** If no mask prompt is provided, a learned, neutral "no mask" embedding is added to each spatial location of the image embedding.

**3. Mask Decoder**

* **Processing the Combined Embedding:** The mask decoder takes the fused embedding (image + prompt) and generates the final segmentation masks. This decoder network learns to interpret the combined information from the image and whatever prompt was provided.

**Why this Approach?**

* **Efficiency:** Element-wise addition is computationally very efficient, allowing for fast and flexible prompting. 
* **Adaptability:** The same mechanism works for different prompt types (points, boxes, masks, text), highlighting SAM's versatility.
* **Informative Fusion:** By combining information directly at the embedding level, SAM forces its learning process to find meaningful relationships between visual features and linguistic concepts.

Refer: https://learnopencv.com/segment-anything/