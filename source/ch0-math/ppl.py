from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("ABC is a startup based in New York City and Paris", return_tensors = "pt")
res = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"])
loss = res.loss
ppl = torch.exp(loss)
print(ppl)

#Output: 29.48

inputs_wiki_text = tokenizer("Generative Pretrained Transformer is an opensource artificial intelligence created by OpenAI in February 2019", return_tensors = "pt")
loss = model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl)

#Output: 211.81