from transformers import RobertaTokenizer, RobertaModel
from hiq.vis import print_model
from util import WORDS, CONTEXTS, words_contexts_for_bpe, run
"""
🌳 OPTModel<all params:125239296>
└── OPTDecoder(decoder)
    ├── Embedding(embed_tokens)|weight[50272,768]
    ├── OPTLearnedPositionalEmbedding(embed_positions)|weight[2050,768]
    ├── LayerNorm(final_layer_norm)|weight[768]|bias[768]
    └── ModuleList(layers)
        └── 💠 OPTDecoderLayer(0-11)<🦜:7087872x12>
            ┣━━ OPTAttention(self_attn)
            ┃   ┗━━ 💠 Linear(k_proj,v_proj,q_proj,out_proj)<🦜:590592x4>|weight[768,768]|bias[768]
            ┣━━ 💠 LayerNorm(self_attn_layer_norm,final_layer_norm)<🦜:1536x2>|weight[768]|bias[768]
            ┣━━ Linear(fc1)|weight[3072,768]|bias[3072]
            ┗━━ Linear(fc2)|weight[768,3072]|bias[768]
"""


if __name__ == "__main__":
    words, contexts = words_contexts_for_bpe(WORDS, CONTEXTS)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    print_model(model)
    run(model, tokenizer, words, contexts, "Roberta")
