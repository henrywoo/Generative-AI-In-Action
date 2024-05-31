import torch
from hiq import read_file
from util import plot_attention_map_FLFH, plot_attention_rank
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

def run(sentence, model, tokenizer, draw_attention_map=False, check_attention_rank=False):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    # Get the attention weights
    outputs = model(**inputs)
    attention = outputs.attentions
    if draw_attention_map:
        plot_attention_map_FLFH(tokenizer, attention, inputs, "CODEBERT", figsize=min(len(sentence), 16))
    if check_attention_rank:
        plot_attention_rank(attention, "codebert", "codebert_attn_map_rank.png")

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base", output_attentions=True)

    sentence = "def max(a,b): if a>b: return a else return b"
    run(sentence, model, tokenizer, draw_attention_map=True, check_attention_rank=True)


    sentence = read_file("Sample.java", by_line=False)
    run(sentence, model, tokenizer, draw_attention_map=False, check_attention_rank=True)



