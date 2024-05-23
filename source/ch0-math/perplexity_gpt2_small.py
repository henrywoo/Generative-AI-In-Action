from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

"""
41.344261169433594
90.83582305908203
231.3223876953125
"""

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

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