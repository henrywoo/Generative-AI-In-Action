from transformers import GPT2Tokenizer, GPT2Model
import torch
from hiq.vis import print_model
from util import WORDS, CONTEXTS, words_contexts_for_bpe, run
"""
🌳 GPT2Model<all params:124439808>
├── Embedding(wte)|weight[50257,768]
├── Embedding(wpe)|weight[1024,768]
├── ModuleList(h)
│   └── 💠 GPT2Block(0-11)<🦜:7087872x12>
│       ┣━━ 💠 LayerNorm(ln_1,ln_2)<🦜:1536x2>|weight[768]|bias[768]
│       ┣━━ GPT2Attention(attn)
│       ┃   ┣━━ Conv1D(c_attn)|weight[768,2304]|bias[2304]
│       ┃   ┗━━ Conv1D(c_proj)|weight[768,768]|bias[768]
│       ┗━━ GPT2MLP(mlp)
│           ┣━━ Conv1D(c_fc)|weight[768,3072]|bias[3072]
│           ┗━━ Conv1D(c_proj)|weight[3072,768]|bias[768]
└── LayerNorm(ln_f)|weight[768]|bias[768]
"""


if __name__ == "__main__":
    words, contexts = words_contexts_for_bpe(WORDS, CONTEXTS)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    print_model(model)
    run(model, tokenizer, words, contexts, "GPT2")