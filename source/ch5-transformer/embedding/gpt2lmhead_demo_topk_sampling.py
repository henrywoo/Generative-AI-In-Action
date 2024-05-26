from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from hiq.vis import print_model

"""
🌳 GPT2LMHeadModel<all params:124439808>
├── GPT2Model(transformer)
│   ├── Embedding(wte)|weight[50257,768]
│   ├── Embedding(wpe)|weight[1024,768]
│   ├── ModuleList(h)
│   │   └── 💠 GPT2Block(0-11)<🦜:7087872x12>
│   │       ┣━━ 💠 LayerNorm(ln_1,ln_2)<🦜:1536x2>|weight[768]|bias[768]
│   │       ┣━━ GPT2Attention(attn)
│   │       ┃   ┣━━ Conv1D(c_attn)|weight[768,2304]|bias[2304]
│   │       ┃   ┗━━ Conv1D(c_proj)|weight[768,768]|bias[768]
│   │       ┗━━ GPT2MLP(mlp)
│   │           ┣━━ Conv1D(c_fc)|weight[768,3072]|bias[3072]
│   │           ┗━━ Conv1D(c_proj)|weight[3072,768]|bias[768]
│   └── LayerNorm(ln_f)|weight[768]|bias[768]
└── Linear(lm_head)|weight[50257,768]
"""

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
print_model(model)

# Input text
text = "The capital of California is"
max_length = 50  # maximum length of the generated sequence
k = 50  # Top-k sampling

# Tokenize the input text
encoded_input = tokenizer(text, return_tensors='pt')

# Generate until EOS token is reached
generated_text = text

for _ in range(max_length):
    # Get the logits for the next token prediction
    with torch.no_grad():
        output = model(**encoded_input)

    # Get the logits of the last token in the input sequence
    logits = output.logits
    last_token_logits = logits[:, -1, :]

    # Apply top-k sampling
    top_k_logits, top_k_indices = torch.topk(last_token_logits, k)
    probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
    predicted_token_id = torch.multinomial(probabilities, 1).item()

    # Decode the predicted token
    predicted_token = tokenizer.decode(predicted_token_id)

    # Append the predicted token to the generated text
    generated_text += tokenizer.decode([predicted_token_id])

    # Break if the EOS token is generated
    if predicted_token_id == tokenizer.eos_token_id:
        break

    # Update the input with the new token
    encoded_input = tokenizer(generated_text, return_tensors='pt')

print(f"Generated text: '{generated_text}'")
