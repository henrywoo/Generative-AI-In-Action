import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

# Load DeepSeek Coder Model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    trust_remote_code=True,
    torch_dtype=torch.float16  # Use float16 for most GPUs, change to bfloat16 for newer ones
).cuda()


# Generation Function
def generate_one_completion(prompt: str, temperature=0.2):  # Slightly higher temperature for more diversity
    messages = [{'role': 'user', 'content': prompt}]  # Chat format for better prompts
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        inputs,
        max_new_tokens=512,  # Adjust if necessary
        do_sample=True,
        top_p=0.95,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    completion = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return completion


# HumanEval
problems = read_problems()
num_samples_per_task = 3  # You can adjust this number as needed
samples = []
for task_id in tqdm(problems):
    for _ in range(num_samples_per_task):  # Generate multiple samples for better evaluation
        completion = generate_one_completion(problems[task_id]["prompt"])
        samples.append(dict(task_id=task_id, completion=completion))

write_jsonl("deepseek_samples.jsonl", samples)

# Evaluation in the HumanEval Sandbox
# Run `evaluate_functional_correctness deepseek_samples.jsonl`
