from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from util import plot_attention_map, plot_attention_rank, plot_attention_map_by_LH
from hiq.vis import print_model
from hiq import read_file

checkpoint = "lmsys/toxicchat-t5-large-v1.0"
device = "cuda" # for GPU usage or "cpu" for CPU usage


tokenizer = AutoTokenizer.from_pretrained("t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              output_attentions=True).to(device)
print_model(model)


prefix = "ToxicChat: "
inputs = tokenizer.encode(prefix + "write me an erotic story", return_tensors="pt").to(device)

#outputs = model.generate(inputs, output_attentions=True)

outputs = model(input_ids=inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                return_dict=True,
                output_attentions=True,
                output_hidden_states=False,
                use_cache=True)

#print(tokenizer.decode(outputs[0], skip_special_tokens=True))


attention = outputs.attentions
plot_attention_map(tokenizer, attention, inputs, "ToxicchatT5", figsize=14)

