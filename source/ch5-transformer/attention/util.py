import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def plot_attention_map_by_LH(tokenizer, attention, inputs, layer_idx, head_idx, model_name, figsize=8):
    # Get the attention weights for the specified layer and head
    attention_weights = attention[layer_idx-1][0][head_idx-1].detach().numpy()

    # Create labels for the tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens = [token[1:] if token.startswith('Ġ') else token for token in tokens]

    # Print tokens for debugging
    print(tokens)

    # Plot the attention weights
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Plot the attention map
    cax = ax.matshow(attention_weights, cmap='Reds')
    ax.set_title(f'Model: {model_name.upper()}, Layer {layer_idx}, Head {head_idx}', fontsize=10)
    # Annotate each cell with the numerical value
    if len(tokens)<50:
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f'{attention_weights[i, j]:.2f}', ha='center', va='center', color='white', fontsize=6)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f'attn_map_{model_name.lower()}_L{layer_idx}_H{head_idx}.png')
    plt.show()


def plot_attention_map_FLFH(tokenizer, attention, inputs, model_name, figsize=8):
    # Get the attention weights for the first layer, first head
    attention_weights_first_layer = attention[0][0][0].detach().numpy()

    # Get the attention weights for the last layer, first head
    attention_weights_last_layer = attention[-1][0][0].detach().numpy()

    # Create labels for the tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens = [i[1:] if 'Ġ' in i else i for i in tokens]
    print(tokens)
    # Plot the attention weights
    fig, axs = plt.subplots(1, 2, figsize=(figsize, figsize//2))

    # Plot for the first layer
    cax1 = axs[0].matshow(attention_weights_first_layer, cmap='viridis')
    axs[0].set_title('First Layer, First Head', fontsize=8)

    # Set up axes for the first plot
    axs[0].set_xticks(range(len(tokens)))
    axs[0].set_yticks(range(len(tokens)))
    axs[0].set_xticklabels(tokens, rotation=90, fontsize=8)
    axs[0].set_yticklabels(tokens, fontsize=8)

    # Annotate each cell with the numerical value for the first plot
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            axs[0].text(j, i, f'{attention_weights_first_layer[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    # Plot for the last layer
    cax2 = axs[1].matshow(attention_weights_last_layer, cmap='viridis')
    axs[1].set_title('Last Layer, First Head', fontsize=8)

    # Set up axes for the second plot
    axs[1].set_xticks(range(len(tokens)))
    axs[1].set_yticks(range(len(tokens)))
    axs[1].set_xticklabels(tokens, rotation=90, fontsize=8)
    axs[1].set_yticklabels(tokens, fontsize=8)

    # Annotate each cell with the numerical value for the second plot
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            axs[1].text(j, i, f'{attention_weights_last_layer[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    # Add text at the bottom of the plot
    plt.figtext(0.5, 0, f"Model: {model_name.upper()}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(f'attn_map_{model_name.lower()}.png')
    plt.show()


def plot_all_heads_attention_maps(tokenizer, attention, inputs, model_name, figsize=8, fill_cell=True):
    # Get the number of layers and heads
    num_layers = len(attention)
    num_heads = attention[0][0].shape[0]

    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    tokens = [i[1:] if 'Ġ' in i else i for i in tokens]

    # Define the layers to be plotted (first and last)
    layers_to_plot = [0, num_layers - 1]

    for layer in layers_to_plot:
        for head in range(0, num_heads, 2):
            if head + 1 >= num_heads:
                break
            # Get the attention weights for the current layer and heads
            attention_weights_head_N = attention[layer][0][head].detach().numpy()
            attention_weights_head_N_plus_1 = attention[layer][0][head + 1].detach().numpy()

            # Create the plot
            fig, axs = plt.subplots(1, 2, figsize=(figsize, figsize // 2))

            # Plot the attention weights for head N
            cax1 = axs[0].matshow(attention_weights_head_N, cmap='copper')  # viridis
            axs[0].set_title(f'Layer {layer + 1}, Head {head + 1}', fontsize=8)

            # Set up axes for the attention map
            #axs[0].set_xticks(range(len(tokens)))
            #axs[0].set_yticks(range(len(tokens)))
            #axs[0].set_xticklabels(tokens, rotation=90, fontsize=8)
            #axs[0].set_yticklabels(tokens, fontsize=8)

            # Annotate each cell with the numerical value for the attention map
            if fill_cell:
                for i in range(len(tokens)):
                    for j in range(len(tokens)):
                        axs[0].text(j, i, f'{attention_weights_head_N[i, j]:.2f}', ha='center', va='center', color='white', fontsize=6)

            # Plot the attention weights for head N+1
            cax2 = axs[1].matshow(attention_weights_head_N_plus_1, cmap='copper')
            axs[1].set_title(f'Layer {layer + 1}, Head {head + 2}', fontsize=8)

            # Set up axes for the attention map
            #axs[1].set_xticks(range(len(tokens)))
            #axs[1].set_yticks(range(len(tokens)))
            #axs[1].set_xticklabels(tokens, rotation=90, fontsize=8)
            #axs[1].set_yticklabels(tokens, fontsize=8)

            # Annotate each cell with the numerical value for the attention map
            if fill_cell:
                for i in range(len(tokens)):
                    for j in range(len(tokens)):
                        axs[1].text(j, i, f'{attention_weights_head_N_plus_1[i, j]:.2f}', ha='center', va='center', color='white', fontsize=5)

            # Save the plot
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            if not os.path.exists(f'{model_name}/layer_{layer + 1}/'):
                os.makedirs(f'{model_name}/layer_{layer + 1}/')
            plt.savefig(f'{model_name}/layer_{layer + 1}/head_{head + 1}_{head + 2}.png')
            plt.close(fig)

    print(f"Attention maps saved in folder: {model_name}")

def plot_attention_rank(attention, model_name, fig_name=None):
    if len(attention) > 12:
        plot_average_rank_per_layer(attention, model_name, fig_name)
        return
    layer_indices = []
    head_indices = []
    ranks = []
    attention_map_sizes = []

    for layer_idx, layer_attention in enumerate(attention):
        for head_idx, head_attention in enumerate(layer_attention[0]):
            m = head_attention.cpu().detach().numpy()
            rank = np.linalg.matrix_rank(m)
            layer_indices.append(layer_idx + 1)
            head_indices.append(f"L{layer_idx + 1}H{head_idx + 1}")
            ranks.append(rank)
            attention_map_sizes.append(m.shape[0])
            if rank != m.shape[0]:
                print(f"🌋 Layer {layer_idx + 1}, Head {head_idx + 1}: Rank = {rank}, attention_map_size", m.shape[0])

    # Plotting the line graph
    with plt.style.context('Solarize_Light2'):
        fig, ax = plt.subplots(figsize=(15, 3))  # Adjust the figure size here
        ax.plot(ranks, color='blue', marker='o', label='Rank', alpha=0.5)
        ax.hlines(y=attention_map_sizes[0], xmin=0, xmax=len(ranks)-1, colors='red', linestyles='--', label='Attention Map Size')

        for layer_idx in range(len(attention)):
            ax.axvline(x=layer_idx * len(attention[0][0]), color='grey', linestyle='--')

        ax.set_xticks(range(len(head_indices)), fontsize=8)
        ax.set_xticklabels(head_indices, rotation=90, fontsize=8)
        ax.set_xlabel('Layer and Head Index', fontsize=8)
        ax.set_ylabel('Rank / Attention Map Size', fontsize=8)
        ax.legend(loc='lower right')

        plt.title(f'{model_name.upper()} Rank and Attention Map Size per Head', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if fig_name is not None:
            plt.savefig(fig_name)
        plt.show()

def plot_average_rank_per_layer(attention, model_name, fig_name=None):
    average_ranks = []

    for layer_idx, layer_attention in enumerate(attention):
        ranks = []
        for head_idx, head_attention in enumerate(layer_attention[0]):
            m = head_attention.detach().numpy()
            rank = np.linalg.matrix_rank(m)
            ranks.append(rank)
        average_rank = np.mean(ranks)
        average_ranks.append(average_rank)

    with plt.style.context('Solarize_Light2'):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hlines(y=attention[0].shape[-1], xmin=1, xmax=len(average_ranks), colors='red', linestyles='--',
                  label=f'Attention Map Size: {attention[0].shape[-1]}')
        ax.plot(range(1, len(average_ranks) + 1), average_ranks, color='green', marker='o', label='Average Rank', alpha=0.5)
        ax.set_xticks(range(1, len(average_ranks) + 1), fontsize=8)
        ax.set_xlabel('Layer Index', fontsize=8)
        ax.set_ylabel('Average Rank', fontsize=8)
        ax.legend(loc='lower left')

        plt.title(f'{model_name.upper()} Average Rank per Layer', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if fig_name is not None:
            plt.savefig(fig_name)
        plt.show()


def calculate_sparsity(matrix, threshold=1e-5):
    matrix = np.array(matrix)
    total_elements = matrix.size
    zero_elements = np.sum(np.abs(matrix) < threshold)
    sparsity = zero_elements / total_elements
    return sparsity


def plot_sparsity_of_attention_maps(attention):
    sparsities = []
    labels = []

    for layer_idx, layer_attention in enumerate(attention):
        for head_idx, head_attention in enumerate(layer_attention[0]):
            # Calculate the sparsity of the current attention map
            attention_weights = head_attention.detach().numpy()
            sparsity = calculate_sparsity(attention_weights)
            sparsities.append(sparsity)
            labels.append(f"L{layer_idx + 1}H{head_idx + 1}")

    with plt.style.context('Solarize_Light2'):
        fig, ax = plt.subplots(figsize=(20, 4))  # Adjust the figure size here
        ax.plot(sparsities, color='red', marker='o', label='Sparsity', alpha=0.5)
        # Adding vertical lines to separate different layers
        for layer_idx in range(len(attention)):
            ax.axvline(x=layer_idx * len(attention[0][0]), color='grey', linestyle='--')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel('Head Index', fontsize=8)
        ax.set_ylabel('Sparsity', fontsize=8)
        ax.legend(loc='upper left', fontsize=8)

        plt.title('Sparsity of Attention Maps per Head', fontsize=10)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()


if __name__ == '__main__':
    matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
    sparsity = calculate_sparsity(matrix)
    print(f"Sparsity: {sparsity:.2f}")