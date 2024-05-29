from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from hiq.vis import print_model

"""
🌳 GPTNeoForCausalLM<all params:1315575808>
├── GPTNeoModel(transformer)
│   ├── Embedding(wte)|weight[50257,2048]
│   ├── Embedding(wpe)|weight[2048,2048]
│   ├── ModuleList(h)
│   │   └── 💠 GPTNeoBlock(0-23)<🦜:50352128x24>
│   │       ┣━━ 💠 LayerNorm(ln_1,ln_2)<🦜:4096x2>|weight[2048]|bias[2048]
│   │       ┣━━ GPTNeoAttention(attn)
│   │       ┃   ┗━━ GPTNeoSelfAttention(attention)
│   │       ┃       ┣━━ 💠 Linear(k_proj,v_proj,q_proj)<🦜:4194304x3>|weight[2048,2048]
│   │       ┃       ┗━━ Linear(out_proj)|weight[2048,2048]|bias[2048]
│   │       ┗━━ GPTNeoMLP(mlp)
│   │           ┣━━ Linear(c_fc)|weight[8192,2048]|bias[8192]
│   │           ┗━━ Linear(c_proj)|weight[2048,8192]|bias[2048]
│   └── LayerNorm(ln_f)|weight[2048]|bias[2048]
└── Linear(lm_head)|weight[50257,2048]
"""

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

print_model(model)

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
