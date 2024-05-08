import torch
from transformers import AutoModel, AutoTokenizer

# Load the LLaVA model and tokenizer
model_name = "haotian-liu/llava-v1.5"  # Or a different LLaVA model variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample text input
text = "A fluffy dog playing in the park."

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt")

# Get text embeddings
with torch.no_grad():
    text_embeddings = model(**inputs).last_hidden_state[:, 0, :] # CLS token representation

# Use the text embeddings for your downstream tasks
print(text_embeddings.shape)  # Example: torch.Size([1, 768])
