from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
tokenizer = AutoTokenizer.from_pretrained("Qwen/CodeQwen1.5-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/CodeQwen1.5-7B",
                                             device_map="auto",
                                             output_attentions=True).eval()

# tokenize the input into tokens
input_text = "#write a quick sort algorithm"
input_text = """Write the next several lines of the following code.
Don't return a preamble or suffix, just the code.

        testBatch(new Double[] { 1.0, 0.0, 123.1234, }, PUnsignedDouble.INSTANCE);

        testBatch(
            new Long[] { 1L, 0L, -1L, Long.MAX_VALUE, Long.MIN_VALUE, 123L, -123L,
                    random.nextLong(), random.nextLong() }, PLong.INSTANCE);

        testBatch(new Long[] { 1L, 0L, Long.MAX_VALUE, 123L }, PUnsignedLong.INSTANCE);

        testBatch(
            new Integer[] { 1, 0, -1, Integer.MAX_VALUE, Integer.MIN_VALUE, 123, -123,"""
model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

# Use `max_new_tokens` to control the maximum output length.
output = model.generate(model_inputs.input_ids,
                               #max_new_tokens=51200,
                               #do_sample=False,
                               #output_attentions=True
                        )[0]
#attentions = output.attentions
generated_ids = output.sequences[:, len(model_inputs.input_ids[0]):]
# The generated_ids include prompt_ids, so we only need to decode the tokens after prompt_ids.
output_text = tokenizer.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Prompt: {input_text}\n\nGenerated text: {output_text}")

