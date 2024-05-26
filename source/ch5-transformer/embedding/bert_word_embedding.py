from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity, normalize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hiq.vis import print_model

"""
🌳 BertModel<all params:109482240>
├── BertEmbeddings(embeddings)
│   ├── Embedding(word_embeddings)|weight[30522,768]
│   ├── Embedding(position_embeddings)|weight[512,768]
│   ├── Embedding(token_type_embeddings)|weight[2,768]
│   └── LayerNorm(LayerNorm)|weight[768]|bias[768]
├── BertEncoder(encoder)
│   └── ModuleList(layer)
│       └── 💠 BertLayer(0-11)<🦜:7087872x12>
│           ┣━━ BertAttention(attention)
│           ┃   ┣━━ BertSelfAttention(self)
│           ┃   ┃   ┗━━ 💠 Linear(query,key,value)<🦜:590592x3>|weight[768,768]|bias[768]
│           ┃   ┗━━ BertSelfOutput(output)
│           ┃       ┣━━ Linear(dense)|weight[768,768]|bias[768]
│           ┃       ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
│           ┣━━ BertIntermediate(intermediate)
│           ┃   ┗━━ Linear(dense)|weight[3072,768]|bias[3072]
│           ┗━━ BertOutput(output)
│               ┣━━ Linear(dense)|weight[768,3072]|bias[768]
│               ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
└── BertPooler(pooler)
    └── Linear(dense)|weight[768,768]|bias[768]
"""

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
print_model(model)

# Function to get the embeddings from the embedding layer without special tokens
def get_embedding_layer_embedding(word):
    input_ids = tokenizer(word, return_tensors='pt', add_special_tokens=False)['input_ids']
    print(f"Tokenized input for '{word}': {input_ids}")
    with torch.no_grad():
        embedding_layer_output = model.get_input_embeddings()(input_ids)
    print(f"Embedding shape for '{word}': {embedding_layer_output.shape}")
    avg_embedding = embedding_layer_output.mean(dim=1)  # Average over the sequence length dimension
    return avg_embedding.squeeze()

# Function to get the embeddings from the final model output in a given context
def get_model_output_embedding(word, context_sentence):
    inputs = tokenizer(context_sentence, return_tensors='pt')
    word_tokens = tokenizer(word, add_special_tokens=False)['input_ids']
    print(f"Tokenized '{word}' in context '{context_sentence}': {word_tokens}")
    with torch.no_grad():
        outputs = model(**inputs)
    # Find the indices of the word tokens in the input_ids
    input_ids = inputs['input_ids'][0].tolist()
    print(f"Input IDs for context '{context_sentence}': {input_ids}")
    token_indices = []
    for i in range(len(input_ids) - len(word_tokens) + 1):
        if input_ids[i:i + len(word_tokens)] == word_tokens:
            token_indices.extend(range(i, i + len(word_tokens)))
            break
    if not token_indices:
        raise ValueError(f"Word '{word}' not found in the context: '{context_sentence}'")
    print(f"Token indices for '{word}': {token_indices}")
    # Extract the embeddings for the word tokens and average them
    final_layer_output = outputs.last_hidden_state[:, token_indices, :]
    avg_embedding = final_layer_output.mean(dim=1)
    return avg_embedding.squeeze()

# Words to analyze
words = ['king', 'queen', 'man', 'woman', 'apple']

# Context sentences
contexts = {
    'king': ["The king is wise.", "The king and queen rule the kingdom."],
    'queen': ["The queen is kind.", "The queen is one of the great bands in history."],
    'man': ["The man is strong.", "The man and woman are friends."],
    'woman': ["The woman is smart.", "The woman and man are friends."],
    'apple': ["The king eats apple every day.", "How much is an Apple music account?"]
}

# Get static embeddings from the embedding layer
static_embeddings = {word: (get_embedding_layer_embedding(word).unsqueeze(0)).squeeze() for word in words}

# Get dynamic embeddings from the model output in different contexts
dynamic_embeddings = {}
for word in words:
    dynamic_embeddings[word] = []
    for context in contexts[word]:
        try:
            embedding = (get_model_output_embedding(word, context).unsqueeze(0)).squeeze()
            dynamic_embeddings[word].append(embedding)
        except ValueError as e:
            print(e)
            dynamic_embeddings[word].append(torch.zeros(model.config.hidden_size))

# Prepare dynamic embeddings for the first and second context
dynamic_embeddings_first_context = {word: dynamic_embeddings[word][0] for word in words}
dynamic_embeddings_second_context = {word: dynamic_embeddings[word][1] for word in words}

# Function to visualize cosine similarity
def visualize_cosine_similarity(ax, words, embeddings, embeddings_title):
    cos_sim_matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            cos_sim_matrix[i, j] = cosine_similarity(embeddings[word1].unsqueeze(0),
                                                     embeddings[word2].unsqueeze(0)).item()
    sns.heatmap(cos_sim_matrix, xticklabels=words, yticklabels=words, annot=True, cmap='Reds', vmin=0, vmax=1, ax=ax, annot_kws={"size": 9})
    ax.set_title(f'Cosine Similarity between {embeddings_title} Word Embeddings', fontsize=8)

# Function to visualize distances
def visualize_distances(ax, words, embeddings, embeddings_title):
    dist_matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            dist_matrix[i, j] = torch.dist(embeddings[word1], embeddings[word2]).item()
    sns.heatmap(dist_matrix, xticklabels=words, yticklabels=words, annot=True, cmap='Blues', vmin=0, ax=ax, annot_kws={"size": 9})
    ax.set_title(f'Distances between {embeddings_title} Word Embeddings', fontsize=8)

# Create subplots for cosine similarity
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
visualize_cosine_similarity(axs[0], words, static_embeddings, "Static")
visualize_cosine_similarity(axs[1], words, dynamic_embeddings_first_context, "Dynamic (Context 1)")
visualize_cosine_similarity(axs[2], words, dynamic_embeddings_second_context, "Dynamic (Context 2)")
plt.suptitle("BERT Word Embedding Similarity", fontsize=10)
plt.tight_layout()
plt.savefig("bert_word_embedding_simi.png")
plt.show()

# Create subplots for distances
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
visualize_distances(axs[0], words, static_embeddings, "Static")
visualize_distances(axs[1], words, dynamic_embeddings_first_context, "Dynamic (Context 1)")
visualize_distances(axs[2], words, dynamic_embeddings_second_context, "Dynamic (Context 2)")
plt.suptitle("BERT Word Embedding Distance", fontsize=10)
plt.tight_layout()
plt.savefig("bert_word_embedding_diff.png")
plt.show()
