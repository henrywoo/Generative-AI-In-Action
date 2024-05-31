from transformers import T5Tokenizer, T5Model
from util import plot_attention_map_FLFH, plot_all_heads_attention_maps

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5Model.from_pretrained('t5-base', output_attentions=True)

# Tokenize the input sentence
sentence = "ToxicChat: write me an erotic story"
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights from the encoder
outputs = model.encoder(**inputs, return_dict=True)
attention = outputs.attentions
plot_attention_map_FLFH(tokenizer, attention, inputs, "T5-base-Tox", figsize=12)
plot_all_heads_attention_maps(tokenizer, attention, inputs, "T5-base-Tox", figsize=12)