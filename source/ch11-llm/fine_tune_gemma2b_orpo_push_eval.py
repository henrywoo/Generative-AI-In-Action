import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel


local_model_path ="google/gemma-2b"
active_model = "gemma-2b-fuheng-orpo"
final_checkpoint = active_model

# Reload model in FP16 (instead of NF4)
base_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, final_checkpoint)
model = model.merge_and_unload()

# Save model and tokenizer
#model.save_pretrained(active_model)
#tokenizer.save_pretrained(active_model)

#hf_token = os.getenv("HF_TOKEN")
#model.push_to_hub(active_model, use_temp_dir=False, token=hf_token)
#tokenizer.push_to_hub(active_model, use_temp_dir=False, token=hf_token)

def run_active_model(active_model):
    tokenizer = AutoTokenizer.from_pretrained(active_model, torch_dtype=torch.bfloat16, device_map="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        active_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig
            (
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ))  # You may want to use bfloat16 and/or move to GPU here

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "how can i help you?",
        },
        {"role": "user", "content": "Can you tell me who is the president of USA?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

    tokenized_chat = tokenized_chat.to('cuda')
    outputs = model.generate(tokenized_chat, max_new_tokens=128)
    print(tokenizer.decode(outputs[0]))

run_active_model(active_model)

