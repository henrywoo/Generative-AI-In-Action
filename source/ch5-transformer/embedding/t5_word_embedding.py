from transformers import T5Tokenizer, T5EncoderModel
from hiq.vis import print_model
from util import WORDS, CONTEXTS, run

"""
🌳 T5EncoderModel<all params:109628544>
├── Embedding(shared)|weight[32128,768]
└── T5Stack(encoder)
    ├── Embedding(embed_tokens)|weight[32128,768]
    ├── ModuleList(block)
    │   ├── T5Block(0)
    │   │   └── ModuleList(layer)
    │   │       ├── T5LayerSelfAttention(0)
    │   │       │   ├── T5Attention(SelfAttention)
    │   │       │   │   ├── 💠 Linear(q,k,v,o)<🦜:589824x4>|weight[768,768]
    │   │       │   │   └── Embedding(relative_attention_bias)|weight[32,12]
    │   │       │   └── T5LayerNorm(layer_norm)|weight[768]
    │   │       └── T5LayerFF(1)
    │   │           ├── T5DenseActDense(DenseReluDense)
    │   │           │   ├── Linear(wi)|weight[3072,768]
    │   │           │   └── Linear(wo)|weight[768,3072]
    │   │           └── T5LayerNorm(layer_norm)|weight[768]
    │   └── 💠 T5Block(1-11)<🦜:7079424x11>
    │       ┗━━ ModuleList(layer)
    │           ┣━━ T5LayerSelfAttention(0)
    │           ┃   ┣━━ T5Attention(SelfAttention)
    │           ┃   ┃   ┗━━ 💠 Linear(q,k,v,o)<🦜:589824x4>|weight[768,768]
    │           ┃   ┗━━ T5LayerNorm(layer_norm)|weight[768]
    │           ┗━━ T5LayerFF(1)
    │               ┣━━ T5DenseActDense(DenseReluDense)
    │               ┃   ┣━━ Linear(wi)|weight[3072,768]
    │               ┃   ┗━━ Linear(wo)|weight[768,3072]
    │               ┗━━ T5LayerNorm(layer_norm)|weight[768]
    └── T5LayerNorm(final_layer_norm)|weight[768]
"""


if __name__ == "__main__":
    words, contexts = WORDS, CONTEXTS
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5EncoderModel.from_pretrained('t5-base')
    print_model(model)
    run(model, tokenizer, words, contexts, "T5")