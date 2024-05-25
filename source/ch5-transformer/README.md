# Transformer

![](img/transformer.webp)


![](img/transformer_ar.webp)

### How to preserve Token Order in High-Dimensional Space? Why we can add position and token embedding?

Each position has a unique encoding based on sine and cosine functions across different dimensions. Even though individual dimensions repeat periodically, the combination across dimensions remains unique for each position in practical sequence lengths.

The model uses the differences in positional encodings to understand the order. Since the positional encoding changes systematically with the position, even if two words are the same, their combined embeddings will differ due to their different positional encodings, allowing the model to distinguish not only which tokens are present but also their order in the sequence.

In high-dimensional space, the order of tokens is not directly apparent as in a 1D list of tokens. The tokens are encoded into the embeddings through mathematical transformations that integrate both the semantic and positional information. The Transformer uses these embeddings, combined with its attention mechanisms, to interpret and maintain the order of tokens throughout its processing layers. This method effectively retains the order information despite the embeddings being in a space where the original order is not immediately visible. So the order is preserved, but it is just not visible to human.

The order of a token significantly impacts its semantic meaning. In a one-dimensional context, semantic content and positional information seem like distinct concepts, often measured in different units, which makes direct addition non-intuitive. However, in a high-dimensional space, both types of information can be abstracted into similar forms, allowing them to be combined more seamlessly. Some variations might concatenate these vectors instead, but addition is preferred because it maintains the dimensionality and allows the model to directly learn interactions between the word's semantic content and its positional context.

Let's take the example for clarity and consistency in explaining how positional embeddings affect the interpretation of sentences:

#### Sentence Pair

1. "He will promise to mail the package tomorrow."
2. "Tomorrow, he will promise to mail the package."

These sentences illustrate the importance of word placement in defining when an action is scheduled versus when a commitment to the action is made:

- **Sentence 1**: "tomorrow" is associated with when the promise will be acted upon. Here, the promise to mail the package is being planned for today, but the mailing action itself is scheduled for tomorrow.
- **Sentence 2**: By starting with "Tomorrow," this sentence shifts the timing of making the promise itself to tomorrow. Thus, not only the action of mailing the package but also the commitment to it is deferred to the next day.

#### Detailed Analysis of Positional and Semantic Embeddings

- **Semantic Embedding**: Each word, such as "tomorrow," "will promise," and "mail," converts into a high-dimensional vector capturing its inherent meaning. "Tomorrow" still refers to a future time point, "will promise" indicates a future commitment, and "mail" continues to denote the action of sending.

- **Positional Embedding**:
  - In **Sentence 1**, "tomorrow" modifies the commitment related to "mail," implying that while the promise is made now, the fulfillment is set for the next day.
  - In **Sentence 2**, "Tomorrow" adjusts the timeline not only for the action but also for the promise, indicating a full deferral of all related activities to the next day.

- **Model Processing**: In Transformers, these embeddings help the model understand how each word affects others, especially regarding sequence and timing. The self-attention mechanism evaluates the positional relationships, using the differences in embeddings to determine that "tomorrow" in the first sentence affects the action timing, while in the second, it influences when the commitment is made.

#### Outcome and Applications

This distinction in timing, enabled by positional embeddings, allows Transformer models to handle subtleties in language that are crucial across various applications:
- **Dialogue Systems**: Understanding the nuance in planning and promises can significantly affect how responses are generated.
- **Scheduling and Planning Tools**: Accurate interpretation of timings in commitments and actions can aid in more effective scheduling, especially in automated systems that interact with human inputs.
- **Legal and Business Document Analysis**: In contexts where the timing of commitments can have legal or financial implications, correctly interpreting such nuances is essential.

These examples show how Transformers can effectively use positional embeddings to navigate the intricacies of language, providing nuanced understanding and responses based on the context provided by the positioning of words like "tomorrow."

### If my data is a time series data, like stock price, where data itself has time info, is sine and cosine position embedding a good choice?

For time series data like stock prices, using sinusoidal positional embeddings as done in NLP **may not be directly applicable or beneficial**. Instead, consider using time directly as a feature, applying transformations relevant to time series analysis, or experimenting with cyclical encoding methods tailored to the periodicity of your data. Always validate the effectiveness of these approaches based on your specific dataset and task requirements.

Transformers, originally developed for natural language processing tasks, have shown promising results in various domains, including time series data prediction. When considering their application for tasks such as predicting stock prices, there are several advantages and considerations to take into account.

#### Advantages of Using Transformers for Time Series Prediction:

1. **Handling Long Dependencies**: One of the key strengths of Transformers is their ability to handle long-range dependencies. In time series data like stock prices, being able to look back over extended periods to identify trends and patterns can be crucial. Transformers, with their self-attention mechanism, can weigh the importance of earlier points without the decay associated with RNNs and LSTMs.

2. **Parallel Computation**: Unlike RNNs, which process data sequentially, Transformers process all data points simultaneously. This parallel processing capability makes them faster and more efficient when training on large datasets.

3. **Flexibility and Customization**: Transformers can be customized with specific attention mechanisms that might be better suited for time series data. For instance, incorporating dilated attention can help focus on periodic patterns, which are common in financial data.

4. **Feature Integration**: Transformers allow for easy integration of various types of features (e.g., volume, open, close, high, low prices) into the model, treating them as additional dimensions in input embeddings.

#### Considerations and Challenges:

1. **Data Preprocessing**: Time series data requires careful preprocessing and normalization. Unlike text data, time series data might have trends, seasonality, and noise that need to be addressed through techniques like differencing, detrending, or seasonal adjustment.

2. **Overfitting Risk**: Given their complexity and capacity, Transformers can easily overfit on time series data, which is often noisy and non-stationary. Regularization, dropout, and proper training/validation splitting are essential to mitigate this.

3. **Lack of Inherent Temporal Order Processing**: Since Transformers do not inherently process temporal order (as they do not have recurrence), positional encodings (or other methods to incorporate time information) are necessary to maintain the sequence order of data.

4. **Computational Intensity**: Transformers require significant computational resources, especially when dealing with large datasets, which can be a limitation for some practical applications.

#### Applications in Stock Price Prediction:

Given these advantages and challenges, Transformers have been applied successfully in financial contexts, but often with modifications:

- **Temporal Fusion Transformers**: An architecture specifically designed for time series forecasting that combines typical Transformer advantages with a gating mechanism to handle the different types of time series-specific variabilities and dependencies.
- **Incorporating External Data**: Transformers can effectively handle additional inputs like news sentiment, economic indicators, or market conditions, integrating them into the stock price prediction model.

#### Example

Here is a simple example of how you might set up a Transformer model for stock price prediction using Python and PyTorch:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define a simple Transformer model (for illustration purposes)
class StockPriceTransformer(nn.Module):
    def __init__(self, feature_size, num_layers, nhead):
        super(StockPriceTransformer, self).__init__()
        self.embedding = nn.Linear(feature_size, feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return output

# Example usage
feature_size = 10  # e.g., open, high, low, close, volume + other features
num_layers = 2
nhead = 2
model = StockPriceTransformer(feature_size, num_layers, nhead)

# Dummy data
data = torch.rand(100, 30, feature_size)  # 100 samples, 30 time steps
labels = torch.rand(100, 1)  # 100 samples, 1 output per sample
dataset = TensorDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=10)

# Example training loop
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(2):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

#### Conclusion

Transformers can be a powerful tool for time series prediction like stock prices, provided that they are carefully adapted and tuned for the specific characteristics and requirements of financial data. Their ability to capture complex dependencies and integrate diverse types of data makes them an intriguing option for advanced forecasting models.

### What is the pretraining objective in T5?

T5 span-masked language modeling is a pre-training objective used for the T5 (Text-to-Text Transfer Transformer) model. It's a form of self-supervised learning where the model learns by predicting missing parts of its input.

![](img/span_masked.webp)

**How it works:**

1. **Span Masking:** Instead of masking single tokens like in traditional masked language modeling (MLM), T5 masks entire spans of text. These spans can vary in length.
2. **Sentinel Tokens:** Unique tokens called "sentinel tokens" (e.g., <extra_id_0>, <extra_id_1>, etc.) are used to replace the masked spans. Each sentinel token indicates the start of a new masked span.
3. **Prediction Target:** The model's task is to predict the original masked spans in the correct order. The output sequence is a concatenation of the sentinel tokens and the corresponding masked spans.

**Example:**

Input sentence:  "The quick brown fox jumps over the lazy dog."

Masked sentence: "The quick <extra_id_0> jumps <extra_id_1> lazy dog."

Target sequence: "<extra_id_0> brown fox <extra_id_1> over the"

**Why span masking:**

* **Captures Longer Dependencies:** Span masking allows the model to learn relationships between words that are farther apart in a sentence, leading to better understanding of context.
* **More Challenging Task:** Predicting entire spans is more difficult than predicting single tokens, potentially leading to a more robust model.
* **Suitable for Text-to-Text Tasks:** Since T5 is designed for various text-to-text tasks (translation, summarization, etc.), span masking aligns well with the model's architecture and objectives.

**In the T5 model:**

Span-masked language modeling is a key component of T5's pre-training. It's one of the factors that contributes to T5's strong performance on a wide range of natural language processing tasks. The model learns to fill in the blanks, effectively understanding how different parts of a text relate to each other. This knowledge can then be transferred to other tasks by fine-tuning the model.

- T5: a detailed explanation https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51



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


## 📌 Reference

- Transformer升级之路：1、Sinusoidal位置编码追根溯源 https://kexue.fm/archives/8231