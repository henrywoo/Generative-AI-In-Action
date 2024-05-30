from transformers import AutoTokenizer, AutoModelForCausalLM
from util import plot_attention_map, plot_attention_rank
from hiq.vis import print_model
from hiq import read_file

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat",
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

plot_attention_map(tokenizer, attention, inputs, "CodeQWen1.5-7B-Chat", figsize=20)

# Decode the generated text
print(f"outputs.logits.shape: {outputs.logits.shape}")
generated_text = tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze(), skip_special_tokens=True)
print("*"*80)
print(generated_text)

#####################################################################################
sentence = read_file("500.txt", by_line=False)
inputs = tokenizer(sentence, return_tensors='pt')

# Get the attention weights
outputs = model(input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True)
attention = outputs.attentions
plot_attention_rank(attention, "CodeQWen1.5-7B-Chat", "CodeQWen1.5-7B-Chat_attn_map_rank.png")