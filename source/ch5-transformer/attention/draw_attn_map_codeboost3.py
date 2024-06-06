from transformers import AutoTokenizer, AutoModelForCausalLM
from util import plot_attention_map_FLFH, plot_attention_rank, plot_attention_map_by_LH, plot_all_heads_attention_maps
from hiq.vis import print_model
from hiq import read_file


model_path = "/home/fuhwu/workspace/codeboost3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             attn_implementation="eager").eval()
print_model(model)

# Tokenize the input sentence
sentence = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
write a quick sort algorithm.<|im_end|>
<|im_start|>assistant"""

inputs = tokenizer(sentence, return_tensors='pt')

print(f"len(sentence): {len(sentence)}")
print(f"inputs.shape:{inputs['input_ids'].shape}")

outputs = model(input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True)
attention = outputs.attentions

plot_attention_map_FLFH(tokenizer, attention, inputs, "Codeboost3", figsize=20)

# Decode the generated text
print(f"outputs.logits.shape: {outputs.logits.shape}")
generated_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)
print("*"*80)
print(generated_text)

#####################################################################################
sentence = read_file("Sample.java", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True)
attention = outputs.attentions
model_name="Codeboost3"
#plot_attention_rank(attention, "Codeboost3", "Codeboost3_attn_map_rank.png")
plot_all_heads_attention_maps(tokenizer, attention, inputs, model_name, figsize=24, fill_cell=False)
#plot_all_heads_attention_maps(tokenizer, attention, inputs, model_name, start_=180, end_=300, figsize=24, fill_cell=False)
#plot_all_heads_attention_maps(tokenizer, attention, inputs, model_name, start_=200, end_=249, figsize=24, fill_cell=False)
#plot_all_heads_attention_maps(tokenizer, attention, inputs, model_name, start_=0, end_=49, figsize=24, fill_cell=False)

#plot_attention_map_by_LH(tokenizer, attention, inputs, 30, 1, model_name, figsize=32)
#plot_attention_map_by_LH(tokenizer, attention, inputs, 30, 12, model_name, figsize=32)
