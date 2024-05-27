import torch
from unixcoder import UniXcoder
from hiq.vis import print_model
from hiq import read_file

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

def func_name_predict():
    context = """
    def <mask0>(data,file_path):
        data = json.dumps(data)
        with open(file_path, 'w') as f:
            f.write(data)
    """
    tokens_ids = model.tokenize([context],max_length=512,mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
    predictions = model.decode(prediction_ids)
    print([x.replace("<mask0>","").strip() for x in predictions[0]])

def api_recommend():
    context = """
    def write_json(data,file_path):
        data = <mask0>(data)
        with open(file_path, 'w') as f:
            f.write(data)
    """
    tokens_ids = model.tokenize([context], max_length=512, mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
    predictions = model.decode(prediction_ids)
    print([x.replace("<mask0>", "").strip() for x in predictions[0]])

def code_sum():
    context = """
    # <mask0>
    def write_json(data,file_path):
        data = json.dumps(data)
        with open(file_path, 'w') as f:
            f.write(data)
    """
    tokens_ids = model.tokenize([context], max_length=512, mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
    predictions = model.decode(prediction_ids)
    print([x.replace("<mask0>", "").strip() for x in predictions[0]])

def java_import():
    context = '''import java.util.Collections;
<mask0>
import java.util.List;

public class Sample {
    public static void main(String[] args) {
        String password = generateSecureRandomPassword();
        String passwordBlock = """
                               <input
                                    id='password'
                                    type='password'
                                    placeholder='%s'
                                    required
                               />
                               """.formatted(password);
        System.out.println(passwordBlock);
    }
}  
'''
    context = read_file("incomplete_code.txt", by_line=False)
    tokens_ids = model.tokenize([context], max_length=1023, mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=256)
    predictions = model.decode(prediction_ids)
    print([x.replace("<mask0>", "").strip() for x in predictions[0]])

func_name_predict()
api_recommend()
code_sum()
java_import()
