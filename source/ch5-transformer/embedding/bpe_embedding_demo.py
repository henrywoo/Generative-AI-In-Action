from transformers import GPT2Tokenizer, GPT2Model
import torch
from torch.nn.functional import cosine_similarity, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')


# Function to get the embeddings for a word in a given context
def get_word_embedding(word, context_sentence):
    # Tokenize the context sentence
    inputs = tokenizer(context_sentence, return_tensors='pt')
    word_tokens = tokenizer(word, add_special_tokens=False)['input_ids']
    print(f"Tokenized '{word}' in context '{context_sentence}': {word_tokens}")

    # Get the tokenized input ids
    input_ids = inputs['input_ids'][0].tolist()
    print(f"Input IDs for context '{context_sentence}': {input_ids}")

    # Convert input_ids back to tokens and then to a sentence
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(f"Tokens: {tokens}")
    reconstructed_sentence = tokenizer.convert_tokens_to_string(tokens)
    print(f"Reconstructed sentence: '{reconstructed_sentence}'")

    # Find the indices of the word tokens in the input_ids
    token_indices = []
    for i in range(len(input_ids) - len(word_tokens) + 1):
        if input_ids[i:i + len(word_tokens)] == word_tokens:
            token_indices.extend(range(i, i + len(word_tokens)))
            break

    if not token_indices:
        raise ValueError(f"Word '{word}' not found in the context: '{context_sentence}'")
    print(f"Token indices for '{word}': {token_indices}")

    # Get the embeddings for the context sentence
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the embeddings for the word tokens and average them
    final_layer_output = outputs.last_hidden_state[:, token_indices, :]
    avg_embedding = final_layer_output.mean(dim=1)
    return avg_embedding.squeeze()


# Example usage
context_sentence = "The king is wise."
word = " king"
word_embedding = get_word_embedding(word, context_sentence)
print(f"Embedding for '{word}': {word_embedding}")


# Visualize the embeddings
def visualize_embedding(word_embedding):
    plt.figure(figsize=(10, 2))
    sns.heatmap(word_embedding.unsqueeze(0).numpy(), annot=True, cmap='viridis')
    plt.title(f'Embedding for "{word}"')
    plt.show()


visualize_embedding(word_embedding)
