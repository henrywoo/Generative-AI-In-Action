from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_error()  # suppress tokenizer warnings

# Step 1: Load small model (TinyLlama-1.1B)
small_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
small_tokenizer = AutoTokenizer.from_pretrained(small_model_id)
small_model = AutoModelForCausalLM.from_pretrained(
    small_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# Step 2: Load big model (Mistral-7B-Instruct-v0.3)
big_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
big_tokenizer = AutoTokenizer.from_pretrained(big_model_id)
big_model = AutoModelForCausalLM.from_pretrained(
    big_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# Step 3: Prompt
prompt = "The capital of France is"
input_ids_small = small_tokenizer(prompt, return_tensors="pt").input_ids.to(small_model.device)

# Step 4: Small model generates a draft
with torch.no_grad():
    draft_ids = small_model.generate(
        input_ids_small,
        max_new_tokens=5,
        do_sample=False
    )

# Extract newly generated draft tokens
draft_new = draft_ids[0][input_ids_small.shape[-1]:]
draft_text = small_tokenizer.decode(draft_new, skip_special_tokens=True)

# Step 5: Use big tokenizer to re-encode prompt + draft
full_text = prompt + " " + draft_text
input_ids_big = big_tokenizer(full_text, return_tensors="pt").input_ids.to(big_model.device)

# Step 6: Big model verifies draft
with torch.no_grad():
    logits = big_model(input_ids_big).logits  # [1, seq_len, vocab_size]

accepted_ids = []
for i in range(len(draft_new)):
    pos = input_ids_big.shape[1] - len(draft_new) + i - 1
    pred_token = logits[0, pos].argmax().item()
    if pred_token == input_ids_big[0, pos + 1].item():
        accepted_ids.append(pred_token)
    else:
        break

# Step 7: Generate fallback from big model if needed
accepted_len = len(accepted_ids)
if accepted_len < len(draft_new):
    fallback_input_ids = input_ids_big[0, :input_ids_big.shape[1] - len(draft_new) + accepted_len].unsqueeze(0)
    with torch.no_grad():
        fallback_output = big_model.generate(
            fallback_input_ids,
            max_new_tokens=len(draft_new) - accepted_len,
            do_sample=False
        )
    fallback_tokens = fallback_output[0][fallback_input_ids.shape[-1]:]
else:
    fallback_tokens = torch.tensor([], dtype=torch.long, device=big_model.device)

# Step 8: Final output
final_ids = torch.cat([
    input_ids_big[0][:input_ids_big.shape[1] - len(draft_new)],
    torch.tensor(accepted_ids, dtype=torch.long, device=big_model.device),
    fallback_tokens
], dim=0)



final_output = big_tokenizer.decode(final_ids, skip_special_tokens=True)
# Print results in plain text (PyCharm-friendly)
print("=" * 50)
print("Prompt:")
print(prompt)
print("\n🧠 Small Model Draft:")
print(draft_text)
print("\n✅ Accepted Tokens by Big Model:")
print(big_tokenizer.decode(accepted_ids))
print("\n🔁 Fallback Tokens from Big Model:")
print(big_tokenizer.decode(fallback_tokens))
print("\n📝 Final Output:")
print(final_output)
print("=" * 50)


'''import pandas as pd
from IPython.display import display

display(pd.DataFrame({
    "Prompt": [prompt],
    "Draft": [draft_text],
    "Accepted Tokens": [big_tokenizer.decode(accepted_ids)],
    "Fallback Tokens": [big_tokenizer.decode(fallback_tokens)],
    "Final Output": [final_output]
}))'''
