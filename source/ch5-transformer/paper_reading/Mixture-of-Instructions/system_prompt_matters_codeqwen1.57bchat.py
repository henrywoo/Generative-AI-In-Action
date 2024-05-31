from transformers import AutoTokenizer, AutoModelForCausalLM
from hiq.vis import print_model

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat",
                                             device_map="auto",
                                             attn_implementation="eager").eval()
#print_model(model)

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "Write a Java function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(text)
print("*"*80)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    output_attentions=True,
    max_new_tokens=1024
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
