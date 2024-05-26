from transformers import GPT2Tokenizer, GPT2Model
from util import plot_attention_map

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