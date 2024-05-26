import matplotlib.pyplot as plt

def plot_attention_map(tokenizer, attention, inputs, model_name, figsize=8):
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
    plt.savefig(f'attn_map_{model_name.lower()}.png')
    plt.show()