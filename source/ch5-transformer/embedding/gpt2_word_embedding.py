from transformers import GPT2Tokenizer, GPT2Model
from torch.nn.functional import cosine_similarity, normalize
import matplotlib.pyplot as plt
import torch
from hiq.vis import print_model
from util import WORDS, CONTEXTS, words_contexts_for_bpe, line_plot_word_embedding, \
    visualize_distances, visualize_cosine_similarity, get_embedding_layer_embedding
"""
🌳 GPT2Model<all params:124439808>
├── Embedding(wte)|weight[50257,768]
├── Embedding(wpe)|weight[1024,768]
├── ModuleList(h)
│   └── 💠 GPT2Block(0-11)<🦜:7087872x12>
│       ┣━━ 💠 LayerNorm(ln_1,ln_2)<🦜:1536x2>|weight[768]|bias[768]
│       ┣━━ GPT2Attention(attn)
│       ┃   ┣━━ Conv1D(c_attn)|weight[768,2304]|bias[2304]
│       ┃   ┗━━ Conv1D(c_proj)|weight[768,768]|bias[768]
│       ┗━━ GPT2MLP(mlp)
│           ┣━━ Conv1D(c_fc)|weight[768,3072]|bias[3072]
│           ┗━━ Conv1D(c_proj)|weight[3072,768]|bias[768]
└── LayerNorm(ln_f)|weight[768]|bias[768]
"""

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
print_model(model)

# Function to get the embeddings from the final model output in a given context
def get_model_output_embedding(model, word, context_sentence):
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

words, contexts = words_contexts_for_bpe(WORDS, CONTEXTS)

# Get static embeddings from the embedding layer
static_embeddings = {word: (get_embedding_layer_embedding(model, tokenizer, word).unsqueeze(0)).squeeze() for word in words}

# Get dynamic embeddings from the model output in different contexts
dynamic_embeddings = {}
for word in words:
    dynamic_embeddings[word] = []
    for context in contexts[word]:
        try:
            embedding = (get_model_output_embedding(model, word, context).unsqueeze(0)).squeeze()
            dynamic_embeddings[word].append(embedding)
        except ValueError as e:
            print(e)
            dynamic_embeddings[word].append(torch.zeros(model.config.n_embd))

# Prepare dynamic embeddings for the first and second context
dynamic_embeddings_1st_context = {word: dynamic_embeddings[word][0] for word in words}
dynamic_embeddings_2nd_context = {word: dynamic_embeddings[word][1] for word in words}

line_plot_word_embedding(words, dynamic_embeddings_1st_context, "GPT2 Dynamic (Context 1)", plot_ab=True)
line_plot_word_embedding(words, dynamic_embeddings_2nd_context, "GPT2 Dynamic (Context 2)", plot_ab=True)

# Create subplots for cosine similarity
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
visualize_cosine_similarity(axs[0], words, static_embeddings, "Static")
visualize_cosine_similarity(axs[1], words, dynamic_embeddings_1st_context, "Dynamic (Context 1)")
visualize_cosine_similarity(axs[2], words, dynamic_embeddings_2nd_context, "Dynamic (Context 2)")
plt.suptitle("GPT2 Word Embedding Similarity", fontsize=10)
plt.tight_layout()
plt.savefig("gpt2_word_embedding_simi.png")
plt.show()

# Create subplots for distances
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
visualize_distances(axs[0], words, static_embeddings, "Static")
visualize_distances(axs[1], words, dynamic_embeddings_1st_context, "Dynamic (Context 1)")
visualize_distances(axs[2], words, dynamic_embeddings_2nd_context, "Dynamic (Context 2)")
plt.suptitle("GPT2 Word Embedding Distance", fontsize=10)
plt.tight_layout()
plt.savefig("gpt2_word_embedding_diff.png")
plt.show()
