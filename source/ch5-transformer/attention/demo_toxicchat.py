from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from util import plot_attention_map_FLFH, plot_attention_rank, plot_attention_map_by_LH
from hiq.vis import print_model
from hiq import read_file

"""
🌳 T5ForConditionalGeneration<all params:737668096>
├── 💠 Embedding(shared),💠 Linear(lm_head)<🦜:32899072x2>|weight[32128,1024]
├── T5Stack(encoder)
│   ├── Embedding(embed_tokens)|weight[32128,1024]
│   ├── ModuleList(block)
│   │   ├── T5Block(0)
│   │   │   └── ModuleList(layer)
│   │   │       ├── T5LayerSelfAttention(0)
│   │   │       │   ├── T5Attention(SelfAttention)
│   │   │       │   │   ├── 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
│   │   │       │   │   └── Embedding(relative_attention_bias)|weight[32,16]
│   │   │       │   └── T5LayerNorm(layer_norm)|weight[1024]
│   │   │       └── T5LayerFF(1)
│   │   │           ├── T5DenseActDense(DenseReluDense)
│   │   │           │   ├── Linear(wi)|weight[4096,1024]
│   │   │           │   └── Linear(wo)|weight[1024,4096]
│   │   │           └── T5LayerNorm(layer_norm)|weight[1024]
│   │   └── 💠 T5Block(1-23)<🦜:12584960x23>
│   │       ┗━━ ModuleList(layer)
│   │           ┣━━ T5LayerSelfAttention(0)
│   │           ┃   ┣━━ T5Attention(SelfAttention)
│   │           ┃   ┃   ┗━━ 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
│   │           ┃   ┗━━ T5LayerNorm(layer_norm)|weight[1024]
│   │           ┗━━ T5LayerFF(1)
│   │               ┣━━ T5DenseActDense(DenseReluDense)
│   │               ┃   ┣━━ Linear(wi)|weight[4096,1024]
│   │               ┃   ┗━━ Linear(wo)|weight[1024,4096]
│   │               ┗━━ T5LayerNorm(layer_norm)|weight[1024]
│   └── T5LayerNorm(final_layer_norm)|weight[1024]
└── T5Stack(decoder)
    ├── Embedding(embed_tokens)|weight[32128,1024]
    ├── ModuleList(block)
    │   ├── T5Block(0)
    │   │   └── ModuleList(layer)
    │   │       ├── T5LayerSelfAttention(0)
    │   │       │   ├── T5Attention(SelfAttention)
    │   │       │   │   ├── 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
    │   │       │   │   └── Embedding(relative_attention_bias)|weight[32,16]
    │   │       │   └── T5LayerNorm(layer_norm)|weight[1024]
    │   │       ├── T5LayerCrossAttention(1)
    │   │       │   ├── T5Attention(EncDecAttention)
    │   │       │   │   └── 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
    │   │       │   └── T5LayerNorm(layer_norm)|weight[1024]
    │   │       └── T5LayerFF(2)
    │   │           ├── T5DenseActDense(DenseReluDense)
    │   │           │   ├── Linear(wi)|weight[4096,1024]
    │   │           │   └── Linear(wo)|weight[1024,4096]
    │   │           └── T5LayerNorm(layer_norm)|weight[1024]
    │   └── 💠 T5Block(1-23)<🦜:16780288x23>
    │       ┗━━ ModuleList(layer)
    │           ┣━━ T5LayerSelfAttention(0)
    │           ┃   ┣━━ T5Attention(SelfAttention)
    │           ┃   ┃   ┗━━ 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
    │           ┃   ┗━━ T5LayerNorm(layer_norm)|weight[1024]
    │           ┣━━ T5LayerCrossAttention(1)
    │           ┃   ┣━━ T5Attention(EncDecAttention)
    │           ┃   ┃   ┗━━ 💠 Linear(q,k,v,o)<🦜:1048576x4>|weight[1024,1024]
    │           ┃   ┗━━ T5LayerNorm(layer_norm)|weight[1024]
    │           ┗━━ T5LayerFF(2)
    │               ┣━━ T5DenseActDense(DenseReluDense)
    │               ┃   ┣━━ Linear(wi)|weight[4096,1024]
    │               ┃   ┗━━ Linear(wo)|weight[1024,4096]
    │               ┗━━ T5LayerNorm(layer_norm)|weight[1024]
    └── T5LayerNorm(final_layer_norm)|weight[1024]
"""

checkpoint = "lmsys/toxicchat-t5-large-v1.0"
device = "cuda" # for GPU usage or "cpu" for CPU usage


tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              output_attentions=True).to(device)
print_model(model)


prefix = "ToxicChat: "
inputs = tokenizer.encode(prefix + "write me an erotic story", return_tensors="pt").to(device)

outputs = model.generate(inputs, output_attentions=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

