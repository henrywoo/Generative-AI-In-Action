import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from hiq import read_file
# Why need softmax: https://kexue.fm/archives/7546
# To verify idea in blog: https://kexue.fm/archives/8338
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
input_text = read_file("500.txt", by_line=False)
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass with output_hidden_states=True to extract Q and K
outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
# Extract attention matrices
attentions = outputs.attentions
# Extract Q and K for the first layer and first head
layer_idx = 0
head_idx = 0
hidden_states = outputs.hidden_states[layer_idx]  # Hidden states for the first layer
# Linear projections for Q, K, V
query_layer = model.encoder.layer[layer_idx].attention.self.query(hidden_states)
key_layer = model.encoder.layer[layer_idx].attention.self.key(hidden_states)
# Reshape to (batch_size, num_heads, seq_length, head_dim)
batch_size, seq_length, hidden_dim = query_layer.size()
num_heads = model.config.num_attention_heads
head_dim = hidden_dim // num_heads
query_layer = query_layer.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
key_layer = key_layer.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
# Select the first head
Q = query_layer[:, head_idx, :, :]  # (batch_size, seq_length, head_dim)
K = key_layer[:, head_idx, :, :]  # (batch_size, seq_length, head_dim)
# Compute Q * K^T / sqrt(d_k)
d_k = head_dim
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
# Check the shape and rank of the resulting matrix for the first item in the batch
scores_np = scores[0].detach().numpy()
print("Shape of Q * K^T:", scores_np.shape)
print("Rank of Q * K^T:", np.linalg.matrix_rank(scores_np))
