import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hiq.vis import print_model

"""
🌳 LlamaForCausalLM<all params:3481766400>
├── LlamaModel(model)
│   ├── Embedding(embed_tokens)|weight[49152,2560]
│   ├── ModuleList(layers)
│   │   └── 💠 LlamaDecoderLayer(0-31)<🦜:104872960x32>
│   │       ┣━━ LlamaSdpaAttention(self_attn)
│   │       ┃   ┗━━ 💠 Linear(q_proj,k_proj,v_proj,o_proj)<🦜:6556160x4>|weight[2560,2560]|bias[2560]
│   │       ┣━━ LlamaMLP(mlp)
│   │       ┃   ┣━━ 💠 Linear(gate_proj,up_proj)<🦜:26214400x2>|weight[10240,2560]
│   │       ┃   ┗━━ Linear(down_proj)|weight[2560,10240]
│   │       ┗━━ 💠 LlamaRMSNorm(input_layernorm,post_attention_layernorm)<🦜:2560x2>|weight[2560]
│   └── LlamaRMSNorm(norm)|weight[2560]
└── Linear(lm_head)|weight[49152,2560]
"""

device = "cuda"
model_path = "ibm-granite/granite-3b-code-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
print_model(model)

# change input text as desired
input_text = "def generate():"
# tokenize the text
input_tokens = tokenizer(input_text, return_tensors="pt")
# transfer tokenized inputs to the device
for i in input_tokens:
    input_tokens[i] = input_tokens[i].to(device)
# generate output tokens
output = model.generate(**input_tokens)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
