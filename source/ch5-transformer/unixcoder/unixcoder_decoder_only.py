import torch
from unixcoder import UniXcoder
from hiq.vis import print_model

"""
🌳 UniXcoder<all params:125929728>
├── RobertaModel(model)
│   ├── RobertaEmbeddings(embeddings)
│   │   ├── Embedding(word_embeddings)|weight[51416,768]
│   │   ├── Embedding(position_embeddings)|weight[1026,768]
│   │   ├── Embedding(token_type_embeddings)|weight[10,768]
│   │   └── LayerNorm(LayerNorm)|weight[768]|bias[768]
│   ├── RobertaEncoder(encoder)
│   │   └── ModuleList(layer)
│   │       └── 💠 RobertaLayer(0-11)<🦜:7087872x12>
│   │           ┣━━ RobertaAttention(attention)
│   │           ┃   ┣━━ RobertaSelfAttention(self)
│   │           ┃   ┃   ┗━━ 💠 Linear(query,key,value)<🦜:590592x3>|weight[768,768]|bias[768]
│   │           ┃   ┗━━ RobertaSelfOutput(output)
│   │           ┃       ┣━━ Linear(dense)|weight[768,768]|bias[768]
│   │           ┃       ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
│   │           ┣━━ RobertaIntermediate(intermediate)
│   │           ┃   ┗━━ Linear(dense)|weight[3072,768]|bias[3072]
│   │           ┗━━ RobertaOutput(output)
│   │               ┣━━ Linear(dense)|weight[768,3072]|bias[768]
│   │               ┗━━ LayerNorm(LayerNorm)|weight[768]|bias[768]
│   └── RobertaPooler(pooler)
│       └── Linear(dense)|weight[768,768]|bias[768]
└── Linear(lm_head)|weight[51416,768]
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("microsoft/unixcoder-base")
model.to(device)
print_model(model)

context = """
def f(data,file_path):
    # write json data into file_path in python language
"""
tokens_ids = model.tokenize([context],max_length=512,mode="<decoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
prediction_ids = model.generate(source_ids, decoder_only=True, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
print(context+predictions[0][0])