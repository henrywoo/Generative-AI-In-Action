from transformers import AutoTokenizer, AutoModelForCausalLM
from util import plot_attention_map
from hiq.vis import print_model

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat",
                                             device_map="auto").eval()
print_model(model)

# Tokenize the input sentence
sentence = "fruit flies like an apple"
inputs = tokenizer(sentence, return_tensors='pt')


outputs = model(input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True)
attention = outputs.attentions



plot_attention_map(tokenizer, attention, inputs, "CodeQWen1.5-7B-Chat")


