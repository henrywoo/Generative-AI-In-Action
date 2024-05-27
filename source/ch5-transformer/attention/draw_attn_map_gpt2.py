from transformers import GPT2Tokenizer, GPT2Model
from util import plot_attention_map, check_attention_rank, plot_attention_map_by_LH, plot_sparsity_of_attention_maps
from hiq import read_file

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)

# Tokenize the input sentence
sentence = "fruit flies like an apple"
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions
plot_attention_map(tokenizer, attention, inputs, "GPT2")

#####################################################################################
sentence = read_file("500.txt", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions
plot_attention_map_by_LH(tokenizer, attention, inputs, 5, 12,"GPT2", figsize=32)
plot_attention_map_by_LH(tokenizer, attention, inputs, 6, 2,"GPT2", figsize=32)
check_attention_rank(attention, "gpt2", "gpt2_attn_map_rank.png")
plot_sparsity_of_attention_maps(attention)