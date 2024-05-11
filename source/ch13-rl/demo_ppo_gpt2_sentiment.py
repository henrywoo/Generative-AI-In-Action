#!/usr/bin/env python
# coding: utf-8

"""
Fine-tune GPT2 to generate positive reviews using a BERT sentiment classifier as a reward function.
"""
import os
import argparse
import torch
from transformers import pipeline, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from trl.core import LengthSampler
from tqdm import tqdm
import pandas as pd
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune GPT2 for generating positive IMDB movie reviews.')
    parser.add_argument('--model_name', type=str, default='lvwerra/gpt2-imdb',
                        help='Model name or path to pre-trained model')
    parser.add_argument('--tokenizer_name', type=str, default='lvwerra/gpt2-imdb',
                        help='Tokenizer name or path')
    parser.add_argument('--bert_model_name', type=str, default='lvwerra/distilbert-imdb',
                        help='BERT model name for sentiment analysis')
    parser.add_argument('--dataset_name', type=str, default='imdb',
                        help='Name of the dataset to use')
    parser.add_argument('--learning_rate', type=float, default=1.41e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train')
    return parser.parse_args()


def load_models_and_tokenizer(config):
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, ref_model, tokenizer


def setup_experiment(config):
    wandb.init()
    device = 0 if torch.cuda.is_available() else 'cpu'  # Simplified device assignment
    sentiment_pipe = pipeline("sentiment-analysis", model=config.bert_model_name, device=device)
    return sentiment_pipe


def build_dataset(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(config.dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(2, 8)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def train_model(config, model, ref_model, tokenizer, sentiment_pipe, dataset, data_collator):
    ppo_config = PPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        log_with="wandb",
    )
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    for epoch in tqdm(range(config.epochs), "Epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            process_batch(batch, ppo_trainer, sentiment_pipe)


output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}


def process_batch(batch, ppo_trainer, sentiment_pipe):
    query_tensors = batch['input_ids']
    response_tensors = []

    # Generate responses using the policy model for each query in the batch
    for query in query_tensors:
        gen_len = output_length_sampler()  # assuming this exists in the config
        generation_kwargs = {
            "max_new_tokens": gen_len,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": ppo_trainer.tokenizer.eos_token_id
        }
        response = ppo_trainer.model.generate(query.unsqueeze(0), **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])

    batch["response"] = [ppo_trainer.tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score using the BERT sentiment analysis pipeline
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Log training statistics
    ppo_trainer.log_stats(stats, batch, rewards)


def model_inspection(ref_model, tuned_model, tokenizer, dataset, device, num_samples=16):
    # Choose a subset of data for inspection
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(num_samples)

    # Prepare data for processing
    query_tensors = [torch.tensor(tokenizer.encode(q)).unsqueeze(0).to(device) for q in df_batch['query']]
    responses_ref, responses_tuned = [], []

    # Generate responses from both models
    for query_tensor in query_tensors:
        gen_kwargs = {
            "max_length": tokenizer.model_max_length,
            "num_beams": 5,
            "early_stopping": True
        }
        response_ref = ref_model.generate(query_tensor, **gen_kwargs)
        response_tuned = tuned_model.generate(query_tensor, **gen_kwargs)
        responses_ref.append(tokenizer.decode(response_ref[0], skip_special_tokens=True))
        responses_tuned.append(tokenizer.decode(response_tuned[0], skip_special_tokens=True))

    # Display the comparisons
    comparisons = pd.DataFrame({
        "Query": df_batch['query'],
        "Response Before": responses_ref,
        "Response After": responses_tuned
    })
    return comparisons


def save_model(model, tokenizer, save_directory, push_to_hub=False):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory, push_to_hub=push_to_hub)
    tokenizer.save_pretrained(save_directory, push_to_hub=push_to_hub)
    print(f"Model and tokenizer have been saved to {save_directory}")


def main():
    args = parse_arguments()
    import huggingface_hub
    huggingface_hub.login(token=os.environ["HF_TOKEN"])
    model, ref_model, tokenizer = load_models_and_tokenizer(args)
    sentiment_pipe = setup_experiment(args)
    dataset = build_dataset(args)
    train_model(args, model, ref_model, tokenizer, sentiment_pipe, dataset, collator)
    save_model(model, tokenizer, "gpt2-imdb-ppo", push_to_hub=True)
    # inspect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tuned_model = model
    ref_model.to(device)
    tuned_model.to(device)
    comparisons = model_inspection(ref_model, tuned_model, tokenizer, dataset, device)
    print(comparisons)


if __name__ == '__main__':
    main()
