from transformers import AutoTokenizer, AutoModelForCausalLM
from util import plot_attention_map
from hiq.vis import print_model

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
print(f"len(sentence): {len(sentence)}")
inputs = tokenizer(sentence, return_tensors='pt')
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

