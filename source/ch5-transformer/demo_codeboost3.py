from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"  # the device to load the model onto

model_path = "/home/fuhwu/workspace/codeboost3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

prompt = """Write the next several lines of the following code.
Don't return a preamble or suffix, just the code.

        testBatch(new Double[] { 1.0, 0.0, 123.1234, }, PUnsignedDouble.INSTANCE);

        testBatch(
            new Long[] { 1L, 0L, -1L, Long.MAX_VALUE, Long.MIN_VALUE, 123L, -123L,
                    random.nextLong(), random.nextLong() }, PLong.INSTANCE);

        testBatch(new Long[] { 1L, 0L, Long.MAX_VALUE, 123L }, PUnsignedLong.INSTANCE);

        testBatch(
            new Integer[] { 1, 10, 100, 1000, 10000, 100000, 1000000,"""
prompt = "Who is founder of Oracle?"
prompt = """
Write a program called solution.py to read integer arguments from the command line and take their product.

Below is an example of how this program should be called from the command line along with the expected output:

python solution.py 5 5 4
100
Do not include any external dependencies in the code, the vanilla language should be enough. Remember to include all necessary imports at the top of the file.
"""
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Open the file for writing
with open("codeboost3.output.txt", "a+") as output_file:
    for i in range(1):
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=5000,
            do_sample=True,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_file.write(response + "\n")
        output_file.write("*" * 40 + f" {i} " + "*" * 40 + "\n")
        print(response)
        print("*" * 40 + f" {i} " + "*" * 40)

print("Responses have been written to codeboost3.output.txt")
