from transformers import AutoTokenizer, AutoModelForCausalLM
from util import plot_attention_map_FLFH, plot_attention_rank, plot_attention_map_by_LH, plot_all_heads_attention_maps
from hiq.vis import print_model
from hiq import read_file

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
#####################################################################################
sentence = read_file("Sample.java", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0], skip_special_tokens=False)
tokens = [i[1:] if 'Ġ' in i else i for i in tokens]
for i, t in enumerate(tokens):
    print(i, t)