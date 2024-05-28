from transformers import BertTokenizer, BertModel
from util import plot_attention_map, plot_attention_rank, plot_attention_map_by_LH
from hiq import read_file

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Tokenize the input sentence
sentence = read_file("500.txt", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(**inputs)
attention = outputs.attentions
plot_attention_rank(attention, "bert-base-uncased", "bert-base-uncased_attn_map_rank.png")
#plot_attention_map(tokenizer, attention, inputs, "BERT_LongSeq", figsize=32)
#plot_attention_map_by_LH(tokenizer, attention, inputs, 12, 12,"BERT", figsize=32)
plot_attention_map_by_LH(tokenizer, attention, inputs, 2, 2,"BERT", figsize=32)