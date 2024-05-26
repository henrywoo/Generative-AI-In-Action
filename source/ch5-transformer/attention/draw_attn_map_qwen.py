from transformers import AutoModelForCausalLM, AutoTokenizer
from util import plot_attention_map

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/CodeQwen1.5-7B-Chat",
    output_attentions=True
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")

# Tokenize the input sentence
sentence = "fruit flies like an apple"
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions
plot_attention_map(tokenizer, attention, inputs, "CodeQwen1.5-7B-Chat")