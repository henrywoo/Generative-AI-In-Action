from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"  # the device to load the model onto

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B", device_map="auto").eval().to(device)

# Define the prompt
prompt = (
    "You are a helpful assistant.\n\n"
    "Question: Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm.\n\n"
    "Answer:"
)

# Tokenize the input
model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the output
generated_ids = model.generate(
    model_inputs.input_ids,
    output_attentions=True,
    max_new_tokens=150  # Control the maximum output length
)

# Decode the generated tokens
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Extract the answer (removing the prompt part)
answer_start = response.find("Answer:") + len("Answer:")
answer = response[answer_start:].strip()

# Print the answer
print("Qwen/CodeQwen1.5-7B's answer:")
print(answer)
