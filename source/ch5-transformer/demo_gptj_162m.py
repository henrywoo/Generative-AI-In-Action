# Warning: This file is deprecated!!!!
from hiq.vis import print_model
from transformers import GPTJForCausalLM, AutoTokenizer
import torch, os
#https://huggingface.co/architext/gptj-162M/discussions/4

"""
🌳 GPTJForCausalLM<all params:162465504>
├── GPTJModel(transformer)
│   ├── Embedding(wte)|weight[50400,768]<f16>
│   ├── ModuleList(h)
│   │   └── 💠 GPTJBlock(0-11)<🦜:7083264x12>
│   │       ┣━━ LayerNorm(ln_1)|weight[768]<f16>|bias[768]<f16>
│   │       ┣━━ GPTJAttention(attn)
│   │       ┃   ┗━━ 💠 Linear(k_proj,v_proj,q_proj,out_proj)<🦜:589824x4>|weight[768,768]<f16>
│   │       ┗━━ GPTJMLP(mlp)
│   │           ┣━━ Linear(fc_in)|weight[3072,768]<f16>|bias[3072]<f16>
│   │           ┗━━ Linear(fc_out)|weight[768,3072]<f16>|bias[768]<f16>
│   └── LayerNorm(ln_f)|weight[768]<f16>|bias[768]<f16>
└── Linear(lm_head)|weight[50400,768]<f16>|bias[50400]<f16>
"""

device = "cuda"
model = GPTJForCausalLM.from_pretrained("architext/gptj-162M", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("architext/gptj-162M")

print_model(model)

prompt = (
    "In a shocking finding, scientists discovered "
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=1000,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)
