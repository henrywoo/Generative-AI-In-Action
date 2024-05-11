from datasets import load_dataset
from transformers import AutoTokenizer

def chatml_format(example):
    message = {"role": "user", "content": example['chosen'][0]['content']}
    # Format instruction
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    # Format chosen answer
    chosen = example['chosen'][1]['content']+tokenizer.eos_token
    # Format rejected answer
    rejected = example['rejected'][1]['content']+tokenizer.eos_token

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load dataset
dataset = load_dataset("argilla/dpo-mix-7k",split="train")
#dataset = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
#dataset = dataset.filter(
#   lambda r:
#       r["status"] != "tie" and
#       r["chosen_score"] >= 5
#       and not r["in_gsm8k_train"]
#)
# Save columns
original_columns = dataset.column_names


# Tokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)

# Print sample
#print(dataset[1])
#Using ORPOTrainer and QLora to tune Gemma 2B
import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
import wandb

from trl import ORPOTrainer
from trl import ORPOConfig

#init env
hf_token = os.getenv("HF_TOKEN")
wandb.login()

#local model path
local_model_path ="google/gemma-2b"
# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    #torch_dtype="auto",
    trust_remote_code=True,
    quantization_config= BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)
model.config.use_cache = False

active_model = "gemma-2b-fuheng-orpo"

torch.cuda.empty_cache()


# Training arguments
training_args = ORPOConfig (
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':True},
    remove_unused_columns=False,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    #max_steps=400,
    num_train_epochs=1,
    save_strategy="no",
    logging_steps=1,
    output_dir=active_model,
    optim="adamw_bnb_8bit",
    warmup_steps=80,
    bf16=True,
    max_prompt_length=256,
    max_length=1024,
    report_to="wandb",
)

# Create DPO trainer
orpo_trainer = ORPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config
)

# Fine-tune model with ORPO
orpo_trainer.train()
final_checkpoint = active_model
orpo_trainer.model.save_pretrained(final_checkpoint)
tokenizer.save_pretrained(final_checkpoint)

#Flush memory
del orpo_trainer, model
gc.collect()
torch.cuda.empty_cache()

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
model.save_pretrained(active_model)
tokenizer.save_pretrained(active_model)

model.push_to_hub(active_model, use_temp_dir=False, token=hf_token)
tokenizer.push_to_hub(active_model, use_temp_dir=False, token=hf_token)

def test_active_model(active_model):
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
        {"role": "user", "content": "what is large language model?"},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

    tokenized_chat = tokenized_chat.to('cuda')
    outputs = model.generate(tokenized_chat, max_new_tokens=128)
    print(tokenizer.decode(outputs[0]))

test_active_model(active_model)

