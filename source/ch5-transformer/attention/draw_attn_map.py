import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Tokenize the input sentence
sentence = "fruit flies like an apple"
#sentence = "British band queen is considered one of the greatest rock bands in history."
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions

# Get the attention weights for the first layer, first head
attention_weights = attention[0][0][0].detach().numpy()

# Create labels for the tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Plot the attention weights
fig, ax = plt.subplots(figsize=(12, 12))
cax = ax.matshow(attention_weights, cmap='viridis')

# Set up axes
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=90)
ax.set_yticklabels(tokens)

# Annotate each cell with the numerical value
for i in range(len(tokens)):
    for j in range(len(tokens)):
        ax.text(j, i, f'{attention_weights[i, j]:.2f}', ha='center', va='center', color='white')

# Display the color bar
#fig.colorbar(cax)

# Show the plot
plt.show()
