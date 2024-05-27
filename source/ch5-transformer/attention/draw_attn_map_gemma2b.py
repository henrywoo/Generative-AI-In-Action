from util import plot_attention_map, plot_attention_rank
from hiq import read_file

# Initialize the tokenizer and model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", output_attentions=True)

# Tokenize the input sentence
sentence = "fruit flies like an apple"
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions

plot_attention_rank(attention, "gemma-2b", "gemma-2b_attn_map_rank_1.png")

plot_attention_map(tokenizer, attention, inputs, "Gemma2b")
#####################################################################################
sentence = read_file("500.txt", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions
plot_attention_rank(attention, "gemma-2b", "gemma-2b_attn_map_rank_2.png")