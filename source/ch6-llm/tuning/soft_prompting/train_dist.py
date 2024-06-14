import torch
from torch.optim import Adam
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from model import SoftPromptTuning
import os

def save_checkpoint(model, optimizer, epoch, loss, filepath='best.pt'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, filepath)

def load_checkpoint(model, optimizer, filepath='best.pt'):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    else:
        return None, None

def train(model, train_dataset, val_dataset, epochs=3, lr=5e-5, checkpoint_path='best.pt', patience=3):
    accelerator = Accelerator()

    # Ensure only the soft prompt embeddings are optimized
    optimizer = Adam([model.soft_prompt_embeddings], lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
    if start_epoch is None:
        start_epoch = 0
        best_val_loss = float('inf')
    
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(start_epoch, epochs), desc="Training Epochs"):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            inputs = model.tokenizer(batch['context'], return_tensors='pt', padding=True, truncation=True)
            labels = model.tokenizer(batch['answers']['text'][0], return_tensors='pt', padding=True, truncation=True).input_ids

            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            labels = labels.to(accelerator.device)

            optimizer.zero_grad()
            with accelerator.autocast():
                outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                logits = outputs.logits[:, model.soft_prompt_len:-1, :].contiguous()

                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                if logits.size(0) != labels.size(0):
                    min_size = min(logits.size(0), labels.size(0))
                    logits = logits[:min_size]
                    labels = labels[:min_size]

                loss = criterion(logits, labels)

            accelerator.backward(loss)
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = model.tokenizer(batch['context'], return_tensors='pt', padding=True, truncation=True)
                labels = model.tokenizer(batch['answers']['text'][0], return_tensors='pt', padding=True, truncation=True).input_ids

                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
                labels = labels.to(accelerator.device)

                outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
                logits = outputs.logits[:, model.soft_prompt_len:-1, :].contiguous()

                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)

                if logits.size(0) != labels.size(0):
                    min_size = min(logits.size(0), labels.size(0))
                    logits = logits[:min_size]
                    labels = labels[:min_size]

                loss = criterion(logits, labels)

                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f"Epoch: {epoch + 1}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")

        # Checkpoint saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    model = SoftPromptTuning()
    
    model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.model.resize_token_embeddings(len(model.tokenizer))

    squad = load_dataset('squad')
    train_data = squad['train']
    val_data = squad['validation']

    train(model, train_data, val_data)

if __name__ == "__main__":
    main()
