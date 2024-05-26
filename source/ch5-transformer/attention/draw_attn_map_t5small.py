from transformers import T5Tokenizer, T5Model
from util import plot_attention_map

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5Model.from_pretrained('t5-small', output_attentions=True)

# Tokenize the input sentence
sentence = "fruit flies like an apple"
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights from the encoder
outputs = model.encoder(**inputs, return_dict=True)
attention = outputs.attentions
plot_attention_map(tokenizer, attention, inputs, "T5-Small")