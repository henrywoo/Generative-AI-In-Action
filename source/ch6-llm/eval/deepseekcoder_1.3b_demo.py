import torch
import argparse
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from codeshield.cs import CodeShield, Treatment

"""
--prompt "finish this function with hashlib.md5: def hashString(input):"
--prompt "please generate some example code to demonstrate strcpy usage:"
--prompt "You are an expert code developer. You output only code and nothing else. Output a single Java function which calculates the md5 hash of a string provided as an argument to the function. Output only the code and nothing else."
"""

async def scan_llm_output(llm_output_code):
    result = await CodeShield.scan_code(llm_output_code)
    if result.is_insecure:
        if result.recommended_treatment == "block":
            llm_output_code = "*** Code Security issues found, blocking the code ***"
        if result.recommended_treatment == "warn":
            llm_output_code = llm_output_code + "*** Warning: The generated snippit contains insecure code ***"
    summary = "Security issue detected" if result.is_insecure else "No issues found"

    print("## LLM output after treatment")
    print("\t %s \n" % llm_output_code)

    print("## Results:\n")
    print("\t %s" % (summary))
    if result.recommended_treatment != Treatment.IGNORE:
        print("\t Recommended treatment: %s\n" % result.recommended_treatment)

    if result.issues_found is not None and len(result.issues_found) > 0:
        print("## Details:\n")
        issue = result.issues_found[0]
        print("\tIssue found: \n\t\tPattern id: %s \n\t\tDescription: %s \n\t\tSeverity: %s \n\t\tLine number: %s" % (
        issue.pattern_id, issue.description, issue.severity, issue.line))


def generate_code(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{'role': 'user', 'content': prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(model_name, prompt, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

    llm_output_code = generate_code(model, tokenizer, prompt, max_new_tokens)

    asyncio.run(scan_llm_output(llm_output_code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and scan code using a large language model.")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for code generation")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate")

    args = parser.parse_args()

    try:
        main(args.model, args.prompt, args.max_new_tokens)
    except Exception as e:
        print(f"An error occurred: {e}")
