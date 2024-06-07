# Experiments

## nanoGTP

```angular2html
🌳 GPT<all params:10745088>
├── ModuleDict(transformer)
│   ├── Embedding(wte)|weight[65,384]
│   ├── Embedding(wpe)|weight[256,384]
│   ├── ModuleList(h)
│   │   └── 💠 Block(0-5)<🦜:1770240x6>
│   │       ┣━━ 💠 LayerNorm(ln_1,ln_2)<🦜:384x2>|weight[384]
│   │       ┣━━ CausalSelfAttention(attn)
│   │       ┃   ┣━━ Linear(c_attn)|weight[1152,384]
│   │       ┃   ┗━━ Linear(c_proj)|weight[384,384]
│   │       ┗━━ MLP(mlp)
│   │           ┣━━ Linear(c_fc)|weight[1536,384]
│   │           ┗━━ Linear(c_proj)|weight[384,1536]
│   └── LayerNorm(ln_f)|weight[384]
└── Linear(lm_head)|weight[65,384]
```

### Preparation

Data: 

https://huggingface.co/datasets/Skylion007/openwebtext


### Training

