import torch
import seaborn as sns
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

plt.style.use('ggplot')

CONTEXTS = {
    'king': ["The king is wise.", "The king and queen rule the kingdom."],
    'queen': ["The queen is the daughter of Prince Edward.", "British band queen is considered one of the greatest rock bands in history."],
    'man': ["The man is strong.", "The man and woman are friends."],
    'woman': ["The woman is smart.", "The woman and man are friends."],
    'apple': ["The king eats apple every day.", "How much is an apple music account?"],
    'flies': ["fruit flies like an apple.", "time flies like an arrow.it's become sort of a colloquialism now. You don't really understand it until you reach your late 30s and early 40s - and I'm sure time will move even faster as I get older."]
}
WORDS = list(CONTEXTS.keys())


def words_for_bpe(ws):
    return [' ' + i for i in ws]


def contexts_for_bpe(ctx):
    r = {}
    for k, v in ctx.items():
        r[' ' + k] = v
    return r


def words_contexts_for_bpe(ws, ctx):
    return words_for_bpe(ws), contexts_for_bpe(ctx)


def get_embedding_layer_embedding(model, tokenizer, word):
    input_ids = tokenizer(word, return_tensors='pt', add_special_tokens=False)['input_ids']
    print(f"Tokenized input for '{word}': {input_ids}")
    with torch.no_grad():
        embedding_layer_output = model.get_input_embeddings()(input_ids)
    print(f"Embedding shape for '{word}': {embedding_layer_output.shape}")
    avg_embedding = embedding_layer_output.mean(dim=1)  # Average over the sequence length dimension
    return avg_embedding.squeeze()



def make_valid_filename(title: str) -> str:
    import re
    title = title.replace(" ", "_").replace("(", "_").replace(")", "_")
    title = re.sub(r'[^\w\.-]', '', title)
    title = re.sub(r'_+', '_', title)
    title = title.strip("_-.")
    return title


def plot_word_pair_embedding(a, b, word1, word2, embeddings_title, ax):
    a = a.squeeze()
    b = b.squeeze()
    if a.shape != b.shape:
        raise ValueError("Vectors a and b must have the same dimensions for plotting.")
    ax.plot(a.numpy(), label=word1, linestyle='-', marker='o', markersize=4, alpha=0.5)
    ax.plot(b.numpy(), label=word2, linestyle='--', marker='x', markersize=4, alpha=0.5)
    ax.set_xlabel('Hidden Dimension', fontsize=10)
    ax.set_ylabel('Embedding Raw Value', fontsize=10)
    ax.set_title(f'{embeddings_title} Word Embeddings({word1} vs {word2})', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.5)


def line_plot_word_embedding(words, embeddings, embeddings_title, plot_ab=False):
    num_plots = len(words)-1  # Number of unique word pairs
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 3))  # Adjust figure size as needed

    plot_index = 0
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j:
                a = embeddings[word1].unsqueeze(0)
                b = embeddings[word2].unsqueeze(0)
                if plot_ab and i == 0:
                    plot_word_pair_embedding(a, b, word1, word2, embeddings_title, axes[plot_index])
                    plot_index += 1

    plt.tight_layout()
    plt.savefig(make_valid_filename(f"{embeddings_title.lower()}_word_embeddings"))
    plt.show()


def visualize_cosine_similarity(ax, words, embeddings, embeddings_title, ignore_outliner=False):
    cos_sim_matrix = np.ones((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i != j:
                a = embeddings[word1].unsqueeze(0)
                b = embeddings[word2].unsqueeze(0)
                if ignore_outliner:
                    sim = cossim_wo_outliers(a, b)
                else:
                    sim = cosine_similarity(a, b).item()
                cos_sim_matrix[i, j] = sim
    sns.heatmap(cos_sim_matrix, xticklabels=words, yticklabels=words,
                annot=True,
                cmap='Reds',
                vmin=0,
                vmax=1,
                ax=ax,
                annot_kws={"size": 9})
    ax.set_title(f'Cosine Similarity between {embeddings_title} Word Embeddings', fontsize=8)


# Function to visualize distances
def visualize_distances(ax, words, embeddings, embeddings_title, ignore_outliner=False):
    dist_matrix = np.zeros((len(words), len(words)))
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if word1 != word2:
                if ignore_outliner:
                    d = dist_wo_outliers(embeddings[word1], embeddings[word2])
                else:
                    d = torch.dist(embeddings[word1], embeddings[word2]).item()
                dist_matrix[i, j] = d
    sns.heatmap(dist_matrix, xticklabels=words, yticklabels=words, annot=True, cmap='Blues', vmin=0, ax=ax,
                annot_kws={"size": 9})
    ax.set_title(f'Distances between {embeddings_title} Word Embeddings', fontsize=8)

def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((data < lower_bound) | (data > upper_bound))
    print("outlier_count:", np.sum(outliers.squeeze().numpy()))
    return np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)


def cossim_wo_outliers(a, b):
    """Use the interquartile range (IQR) method to detect and remove outliers."""
    a_cleaned = remove_outliers(a)
    b_cleaned = remove_outliers(b)
    # Keep only the indices where neither a_cleaned nor b_cleaned is NaN
    valid_indices = ~np.isnan(a_cleaned) & ~np.isnan(b_cleaned)
    a_filtered = a_cleaned[valid_indices]
    b_filtered = b_cleaned[valid_indices]
    if len(a_filtered) == 0 or len(b_filtered) == 0:
        raise ValueError("No valid data points remain after outlier removal.")
    # Compute cosine similarity
    similarity = 1 - cosine(a_filtered, b_filtered)
    return similarity


def dist_wo_outliers(a, b):
    a_cleaned = remove_outliers(a)
    b_cleaned = remove_outliers(b)
    # Keep only the indices where neither a_cleaned nor b_cleaned is NaN
    valid_indices = ~np.isnan(a_cleaned) & ~np.isnan(b_cleaned)
    a_filtered = a_cleaned[valid_indices]
    b_filtered = b_cleaned[valid_indices]
    if len(a_filtered) == 0 or len(b_filtered) == 0:
        raise ValueError("No valid data points remain after outlier removal.")
    # Compute Euclidean distance
    distance = torch.dist(torch.tensor(a_filtered), torch.tensor(b_filtered)).item()
    return distance


def plot_sim_diff(model_name,words, static_embeddings, dynamic_embeddings_1st_context, dynamic_embeddings_2nd_context):
    line_plot_word_embedding(words, dynamic_embeddings_1st_context, f"{model_name.upper()} Dynamic (Context 1)", plot_ab=True)
    line_plot_word_embedding(words, dynamic_embeddings_2nd_context, f"{model_name.upper()} Dynamic (Context 2)", plot_ab=True)
    # Create subplots for cosine similarity
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    visualize_cosine_similarity(axs[0], words, static_embeddings, "Static")
    visualize_cosine_similarity(axs[1], words, dynamic_embeddings_1st_context, "Dynamic (Context 1)")
    visualize_cosine_similarity(axs[2], words, dynamic_embeddings_2nd_context, "Dynamic (Context 2)")
    plt.suptitle(f"{model_name.upper()} Word Embedding Similarity", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_word_embedding_simi.png")
    plt.show()

    # Create subplots for distances
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    visualize_distances(axs[0], words, static_embeddings, "Static")
    visualize_distances(axs[1], words, dynamic_embeddings_1st_context, "Dynamic (Context 1)")
    visualize_distances(axs[2], words, dynamic_embeddings_2nd_context, "Dynamic (Context 2)")
    plt.suptitle(f"{model_name.upper()} Word Embedding Distance", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_word_embedding_diff.png")
    plt.show()

# Function to get the embeddings from the final model output in a given context
def get_model_output_embedding(model, tokenizer, word, context_sentence):
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

def run(model, tokenizer, words, contexts, model_name):
    # Get static embeddings from the embedding layer
    static_embeddings = {word: (get_embedding_layer_embedding(model, tokenizer, word).unsqueeze(0)).squeeze() for word in words}

    # Get dynamic embeddings from the model output in different contexts
    dynamic_embeddings = {}
    for word in words:
        dynamic_embeddings[word] = []
        for context in contexts[word]:
            try:
                embedding = (get_model_output_embedding(model, tokenizer, word, context).unsqueeze(0)).squeeze()
                dynamic_embeddings[word].append(embedding)
            except ValueError as e:
                print(e)
                dynamic_embeddings[word].append(torch.zeros(model.config.hidden_size))

    # Prepare dynamic embeddings for the first and second context
    dynamic_embeddings_1st_context = {word: dynamic_embeddings[word][0] for word in words}
    dynamic_embeddings_2nd_context = {word: dynamic_embeddings[word][1] for word in words}

    plot_sim_diff(model_name, words, static_embeddings, dynamic_embeddings_1st_context, dynamic_embeddings_2nd_context)
