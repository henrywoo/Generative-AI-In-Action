from transformers import AutoTokenizer, AutoModelForCausalLM
from hiq.vis import print_model

"""
🌳 Qwen2ForCausalLM<all params:7250284544>
├── Qwen2Model(model)
│   ├── Embedding(embed_tokens)|weight[92416,4096]
│   ├── ModuleList(layers)
│   │   └── 💠 Qwen2DecoderLayer(0-31)<🦜:202912768x32>
│   │       ┣━━ Qwen2SdpaAttention(self_attn)
│   │       ┃   ┣━━ Linear(q_proj)|weight[4096,4096]|bias[4096]
│   │       ┃   ┣━━ 💠 Linear(k_proj,v_proj)<🦜:2097664x2>|weight[512,4096]|bias[512]
│   │       ┃   ┗━━ Linear(o_proj)|weight[4096,4096]
│   │       ┣━━ Qwen2MLP(mlp)
│   │       ┃   ┣━━ 💠 Linear(gate_proj,up_proj)<🦜:55050240x2>|weight[13440,4096]
│   │       ┃   ┗━━ Linear(down_proj)|weight[4096,13440]
│   │       ┗━━ 💠 Qwen2RMSNorm(input_layernorm,post_attention_layernorm)<🦜:4096x2>|weight[4096]
│   └── Qwen2RMSNorm(norm)|weight[4096]
└── Linear(lm_head)|weight[92416,4096]
"""

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B-Chat", device_map="auto").eval()
print_model(model)

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.
generated_ids = model.generate(
    model_inputs.input_ids,
    output_attentions=True,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
