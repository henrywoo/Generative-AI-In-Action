from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-base", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b-base", trust_remote_code=True).half().cuda()

inputs = tokenizer(["今天天气真不错"], return_tensors="pt").to('cuda')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0].tolist()))