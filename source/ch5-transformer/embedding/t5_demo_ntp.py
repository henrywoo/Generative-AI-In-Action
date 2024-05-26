from transformers import T5Tokenizer, T5ForConditionalGeneration
from hiq.vis import print_model
import torch

def is_weight_tying(model):
    # Access the shared embedding matrix and the final linear layer's weights in the decoder
    shared_embed = model.shared.weight  # T5 uses a shared embedding matrix for both encoder and decoder
    lm_head = model.lm_head.weight  # Final linear layer in the decoder (equivalent to lm_head in GPT-2)

    # Check if they are the same object in memory (pointer equality)
    are_tied = shared_embed is lm_head
    print(f"Are T5 word embeddings and decoder lm_head weights tied? {are_tied}")

    # Check if they have the same values (value equality) - more robust
    are_same_values = torch.allclose(shared_embed, lm_head)
    print(f"Are T5 word embeddings and decoder lm_head values the same? {are_same_values}")


tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")  # t5-small
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
print_model(model)
is_weight_tying(model)

input_text = "The capital of California is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, max_length=50)  # adjust max_length as needed

predicted_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(input_text, predicted_text)