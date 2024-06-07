from datasets import load_dataset
import torch
from tqdm import tqdm
from soft_prompting_training import load_checkpoint
import numpy as np
from model import SoftPromptTuning

# Load SQuAD dataset
squad = load_dataset('squad')
val_data = squad['validation']


def evaluate(model, val_data):
    em_scores = []
    f1_scores = []

    for sample in tqdm(val_data, desc="Evaluating"):
        context = sample['context']
        question = sample['question']
        true_answer = sample['answers']['text'][0]

        input_text = question + " " + context
        input_ids = model.tokenizer(input_text, return_tensors='pt').input_ids

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, model.soft_prompt_len:, :]
            predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
            predicted_text = model.tokenizer.decode(predicted_ids, skip_special_tokens=True)

        # Compute EM and F1
        em = compute_exact_match(predicted_text, true_answer)
        f1 = compute_f1(predicted_text, true_answer)

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)

    return avg_em, avg_f1


def compute_exact_match(pred, truth):
    return int(pred.strip() == truth.strip())


def compute_f1(pred, truth):
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main():
    model = SoftPromptTuning()

    # Add a new pad_token to the tokenizer
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.model.resize_token_embeddings(len(model.tokenizer))

    # Load model from the best checkpoint
    model = load_checkpoint(model, checkpoint_path='best.pt')

    # Load the validation data
    val_data = squad['validation']

    # Perform evaluation
    avg_em, avg_f1 = evaluate(model, val_data)
    print(f"Exact Match (EM): {avg_em:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
