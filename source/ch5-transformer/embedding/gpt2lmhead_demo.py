from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from hiq.vis import print_model

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print_model(model)

def is_weight_tying(model):
    # Access the word embedding matrix (wte) and the lm_head weights
    wte = model.transformer.wte.weight  # Word embeddings
    lm_head = model.lm_head.weight  # Language modeling head weights

    # Check if they are the same object in memory (pointer equality)
    are_tied = wte is lm_head

    print(f"Are GPT-2 word embeddings and lm_head weights tied? {are_tied}")

    # Check if they have the same values (value equality) - more robust
    are_same_values = torch.allclose(wte, lm_head)
    print(f"Are GPT-2 word embeddings and lm_head values the same? {are_same_values}")

is_weight_tying(model)

# Input text
text = "The capital of California is"
max_length = 30
p = 0.9

# Tokenize input
input_ids = tokenizer.encode(text, return_tensors="pt")

# Generate text
output_sequences = model.generate(
    input_ids=input_ids,
    max_length=max_length,
    do_sample=True,
    top_p=p,
    num_return_sequences=3  # Generate a single sequence
)

# Decode generated text
for i in range(3):
    generated_text = tokenizer.decode(output_sequences[i], skip_special_tokens=True)
    print(f"Generated text: '{generated_text}'")
