from transformers import BertTokenizer, BertModel
from hiq.vis import print_model
from util import WORDS, CONTEXTS, run
"""
🌳 BertModel<all params:109482240>
├── BertEmbeddings(embeddings)
│   ├── Embedding(word_embeddings)|weight[30522,768]
│   ├── Embedding(position_embeddings)|weight[512,768]
│   ├── Embedding(token_type_embeddings)|weight[2,768]
│   └── LayerNorm(LayerNorm)|weight[768]|bias[768]
├── BertEncoder(encoder)
│   └── ModuleList(layer)
│       └── 💠 BertLayer(0-11)<🦜:7087872x12>
│           ┣━━ BertAttention(attention)
│           ┃   ┣━━ BertSelfAttention(self)
│           ┃   ┃   ┗━━ 💠 Linear(query,key,value)<🦜:590592x3>|weight[768,768]|bias[768]
│           ┃   ┗━━ BertSelfOutput(output)
│           ┃       ┣━━ Linear(dense)|weight[768,768]|bias[768]
│           ┃       ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
│           ┣━━ BertIntermediate(intermediate)
│           ┃   ┗━━ Linear(dense)|weight[3072,768]|bias[3072]
│           ┗━━ BertOutput(output)
│               ┣━━ Linear(dense)|weight[768,3072]|bias[768]
│               ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
└── BertPooler(pooler)
    └── Linear(dense)|weight[768,768]|bias[768]
"""


if __name__ == "__main__":
    words, contexts = WORDS, CONTEXTS
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print_model(model)
    run(model, tokenizer, words, contexts, "BERT")