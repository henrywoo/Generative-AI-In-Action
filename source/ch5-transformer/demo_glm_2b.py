from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from hiq.vis import print_model
"""
🌳 GLMForConditionalGeneration<all params:1920122880>
└── GLMModel(glm)
    ├── VocabEmbedding(word_embeddings)|weight[50304,2048]<f16>
    └── GLMStack(transformer)
        ├── 💠 Embedding(position_embeddings,block_position_embeddings)<🦜:2099200x2>|weight[1025,2048]<f16>
        ├── ModuleList(layers)
        │   └── 💠 GLMBlock(0-35)<🦜:50358272x36>
        │       ┣━━ 💠 LayerNorm(input_layernorm,post_attention_layernorm)<🦜:4096x2>|weight[2048]<f16>|bias[2048]<f16>
        │       ┣━━ SelfAttention(attention)
        │       ┃   ┣━━ Linear(query_key_value)|weight[6144,2048]<f16>|bias[6144]<f16>
        │       ┃   ┗━━ Linear(dense)|weight[2048,2048]<f16>|bias[2048]<f16>
        │       ┗━━ MLP(mlp)
        │           ┣━━ Linear(dense_h_to_4h)|weight[8192,2048]<f16>|bias[8192]<f16>
        │           ┗━━ Linear(dense_4h_to_h)|weight[2048,8192]<f16>|bias[2048]<f16>
        └── LayerNorm(final_layernorm)|weight[2048]<f16>|bias[2048]<f16>
"""
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-2b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-2b", trust_remote_code=True)
model = model.half().cuda()
model.eval()
print_model(model)

# Inference
inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))

# Training
inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits