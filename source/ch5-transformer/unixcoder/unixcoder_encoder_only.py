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

# Encode maximum function
func = "def f(a,b): if a>b: return a else return b"
tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings, max_func_embedding = model(source_ids)

# Encode minimum function
func = "def get_min(a,b): if a<b: return a else return b"
tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings, min_func_embedding = model(source_ids)

# Encode NL
nl = "a small function f(a,b) to return maximum value of two values with python"
tokens_ids = model.tokenize([nl], max_length=512, mode="<encoder-only>")
source_ids = torch.tensor(tokens_ids).to(device)
tokens_embeddings, nl_embedding = model(source_ids)

print(max_func_embedding.shape)
print(max_func_embedding)

from torch.nn.functional import cosine_similarity

print(cosine_similarity(max_func_embedding, min_func_embedding))
print(cosine_similarity(max_func_embedding, nl_embedding))
print(cosine_similarity(min_func_embedding, nl_embedding))

# Normalize embedding
norm_max_func_embedding = torch.nn.functional.normalize(max_func_embedding, p=2, dim=1)
norm_min_func_embedding = torch.nn.functional.normalize(min_func_embedding, p=2, dim=1)
norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)

max_func_nl_similarity = torch.einsum("ac,bc->ab",norm_max_func_embedding,norm_nl_embedding)
min_func_nl_similarity = torch.einsum("ac,bc->ab",norm_min_func_embedding,norm_nl_embedding)

print(max_func_nl_similarity)
print(min_func_nl_similarity)
