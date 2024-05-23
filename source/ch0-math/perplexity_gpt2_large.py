from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
14.713248252868652
86.92589569091797
214.8533172607422
"""
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

inputs = tokenizer("Albert Einstein was a German-born theoretical physicist.", return_tensors = "pt")
res = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"])
loss = res.loss
ppl = torch.exp(loss)
print(ppl.item())

inputs = tokenizer("Fuheng Wu was a German-born theoretical physicist.", return_tensors = "pt")
res = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"])
loss = res.loss
ppl = torch.exp(loss)
print(ppl.item())


uncommon_text = tokenizer("Fuheng Wu was a German-born theoretical physicist famous for training LLM.", return_tensors = "pt")
loss = model(input_ids = uncommon_text["input_ids"], labels = uncommon_text["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl.item())


