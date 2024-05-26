import torch
from evaluate import load
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Example reference and candidate sentences
references = ["The cat sat on the mat."]
candidates = ["The cat is sitting on the mat.", "There is a cat on the mat."]

# Load BLEU, ROUGE, and METEOR metrics
bleu_metric = load("bleu")
rouge_metric = load("rouge")
meteor_metric = load("meteor")

# Compute BLEU scores
bleu_score1 = bleu_metric.compute(predictions=[candidates[0]], references=references)
bleu_score2 = bleu_metric.compute(predictions=[candidates[1]], references=references)

print(f"BLEU score for candidate1: {bleu_score1['bleu']:.4f}")
print(f"BLEU score for candidate2: {bleu_score2['bleu']:.4f}")

# Prepare data for ROUGE and METEOR
rouge_references = [references[0]] * len(candidates)
rouge_candidates = candidates

# Compute ROUGE scores
rouge_scores = rouge_metric.compute(predictions=rouge_candidates, references=rouge_references)

print(f"ROUGE scores for candidate1 and candidate2: {rouge_scores}")

# Compute METEOR scores
meteor_score1 = meteor_metric.compute(predictions=[candidates[0]], references=references)
meteor_score2 = meteor_metric.compute(predictions=[candidates[1]], references=references)

print(f"METEOR score for candidate1: {meteor_score1['meteor']:.4f}")
print(f"METEOR score for candidate2: {meteor_score2['meteor']:.4f}")

# Perplexity calculation
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    log_likelihood = outputs.loss
    perplexity = torch.exp(log_likelihood)
    return perplexity.item()

# Load GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Compute Perplexity for candidates
perplexity1 = calculate_perplexity(model, tokenizer, candidates[0])
perplexity2 = calculate_perplexity(model, tokenizer, candidates[1])

print(f"Perplexity for candidate1: {perplexity1:.4f}")
print(f"Perplexity for candidate2: {perplexity2:.4f}")
