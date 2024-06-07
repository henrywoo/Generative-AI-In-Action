import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class SoftPromptTuning(torch.nn.Module):
    def __init__(self, model_name='gpt2', soft_prompt_len=10):
        super(SoftPromptTuning, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Initialize soft prompts
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt_embeddings = torch.nn.Parameter(torch.randn(soft_prompt_len, self.model.config.n_embd))

    def forward(self, input_ids, attention_mask=None):
        # Get embeddings for input ids
        input_embeddings = self.model.transformer.wte(input_ids)

        # Concatenate soft prompts to the input embeddings
        soft_prompt_embeddings = self.soft_prompt_embeddings.unsqueeze(0).expand(input_embeddings.size(0), -1, -1)
        inputs_embeds = torch.cat((soft_prompt_embeddings, input_embeddings), dim=1)

        # Adjust attention mask
        if attention_mask is not None:
            soft_prompt_mask = torch.ones((attention_mask.size(0), self.soft_prompt_len), device=attention_mask.device)
            attention_mask = torch.cat((soft_prompt_mask, attention_mask), dim=1)

        # Forward pass through the model
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs
