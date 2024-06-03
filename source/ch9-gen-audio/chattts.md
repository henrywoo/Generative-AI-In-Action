# ChatTTS Model Architecture

- https://github.com/mszulc913/dvae-pytorch
- https://github.com/ituvisionlab/EdVAE
- https://ngwaifoong92.medium.com/introduction-to-vocos-fast-neural-vocoder-e055a27bbf92
- 


## vocos

```angular2html
🌳 Vocos<all params:13531650>
├── MelSpectrogramFeatures(feature_extractor)
│   └── MelSpectrogram(mel_spec)
├── VocosBackbone(backbone)
│   ├── Conv1d(embed)|weight[512,100,7]|bias[512]
│   ├── 💠 LayerNorm(norm,final_layer_norm)<🦜:1024x2>|weight[512]|bias[512]
│   └── ModuleList(convnext)
│       └── 💠 ConvNeXtBlock(0-7)<🦜:1580544x8>|gamma[512]
│           ┣━━ Conv1d(dwconv)|weight[512,1,7]|bias[512]
│           ┣━━ LayerNorm(norm)|weight[512]|bias[512]
│           ┣━━ Linear(pwconv1)|weight[1536,512]|bias[1536]
│           ┗━━ Linear(pwconv2)|weight[512,1536]|bias[512]
└── ISTFTHead(head)
    └── Linear(out)|weight[1026,512]|bias[1026]
```

## Dave

```angular2html
🌳 DVAE<all params:6929800>
├── DVAEDecoder(decoder)
│   ├── Sequential(conv_in)
│   │   ├── Conv1d(0)|weight[128,512,3]|bias[128]
│   │   └── Conv1d(2)|weight[256,128,3]|bias[256]
│   ├── ModuleList(decoder_block)
│   │   └── 💠 ConvNeXtBlock(0-11)<🦜:528384x12>|gamma[256]
│   │       ┣━━ Conv1d(dwconv)|weight[256,1,7]|bias[256]
│   │       ┣━━ LayerNorm(norm)|weight[256]|bias[256]
│   │       ┣━━ Linear(pwconv1)|weight[1024,256]|bias[1024]
│   │       ┗━━ Linear(pwconv2)|weight[256,1024]|bias[256]
│   └── Conv1d(conv_out)|weight[512,256,1]
├── Conv1d(out_conv)|weight[100,512,3]
└── GFSQ(vq_layer)
    └── GroupedResidualFSQ(quantizer)
        └── ModuleList(rvqs)
            └── 💠 ResidualFSQ(0-1)<🦜:4612x2>
                ┣━━ Linear(project_in)|weight[4,512]|bias[4]
                ┣━━ Linear(project_out)|weight[512,4]|bias[512]
                ┗━━ ModuleList(layers)
                    ┗━━ 💠 FSQ(0-1)<🦜:0x2>
```

## GPT

```angular2html
🌳 LlamaModel<all params:213351168>
├── Embedding(embed_tokens)|weight[32000,768]
├── ModuleList(layers)
│   └── 💠 LlamaDecoderLayer(0-19)<🦜:9438720x20>
│       ┣━━ LlamaSdpaAttention(self_attn)
│       ┃   ┗━━ 💠 Linear(q_proj,k_proj,v_proj,o_proj)<🦜:589824x4>|weight[768,768]
│       ┣━━ LlamaMLP(mlp)
│       ┃   ┣━━ 💠 Linear(gate_proj,up_proj)<🦜:2359296x2>|weight[3072,768]
│       ┃   ┗━━ Linear(down_proj)|weight[768,3072]
│       ┗━━ 💠 LlamaRMSNorm(input_layernorm,post_attention_layernorm)<🦜:768x2>|wei
│           ght[768]
└── LlamaRMSNorm(norm)|weight[768]
```


```angular2html
🌳 GPT_warpper<all params:225174402>
├── LlamaModel(gpt)
│   ├── ModuleList(layers)
│   │   └── 💠 LlamaDecoderLayer(0-19)<🦜:9438720x20>
│   │       ┣━━ LlamaSdpaAttention(self_attn)
│   │       ┃   ┗━━ 💠 Linear(q_proj,k_proj,v_proj,o_proj)<🦜:589824x4>|weight[768,768]
│   │       ┣━━ LlamaMLP(mlp)
│   │       ┃   ┣━━ 💠 Linear(gate_proj,up_proj)<🦜:2359296x2>|weight[3072,768]
│   │       ┃   ┗━━ Linear(down_proj)|weight[768,3072]
│   │       ┗━━ 💠 LlamaRMSNorm(input_layernorm,post_attention_layernorm)<🦜:768x2>|weight[768]
│   └── LlamaRMSNorm(norm)|weight[768]
├── ModuleList(emb_code)
│   └── 💠 Embedding(0-3)<🦜:480768x4>|weight[626,768]
├── Embedding(emb_text)|weight[21178,768]
├── ParametrizedLinear(head_text)
│   └── ModuleDict(parametrizations)
│       └── ParametrizationList(weight)|original0[21178,1]|original1[21178,768]
└── ModuleList(head_code)
    └── 💠 ParametrizedLinear(0-3)<🦜:481394x4>
        ┗━━ ModuleDict(parametrizations)
            ┗━━ ParametrizationList(weight)|original0[626,1]|original1[626,768]
```

## DAVE

```angular2html
🌳 DVAE<all params:25920640>
├── DVAEDecoder(decoder)
│   ├── Sequential(conv_in)
│   │   ├── Conv1d(0)|weight[128,384,3]|bias[128]
│   │   └── Conv1d(2)|weight[512,128,3]|bias[512]
│   ├── ModuleList(decoder_block)
│   │   └── 💠 ConvNeXtBlock(0-11)<🦜:2105344x12>|gamma[512]
│   │       ┣━━ Conv1d(dwconv)|weight[512,1,7]|bias[512]
│   │       ┣━━ LayerNorm(norm)|weight[512]|bias[512]
│   │       ┣━━ Linear(pwconv1)|weight[2048,512]|bias[2048]
│   │       ┗━━ Linear(pwconv2)|weight[512,2048]|bias[512]
│   └── Conv1d(conv_out)|weight[384,512,1]
└── Conv1d(out_conv)|weight[100,384,3]
```