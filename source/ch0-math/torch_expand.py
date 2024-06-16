import torch

# Original tensor
latent_tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("Original tensor (latent_tokens):")
print(latent_tokens)

# Expand the tensor
B = 3
expanded_tokens = latent_tokens.unsqueeze(0).expand(B, -1, -1)
print("\nExpanded tensor (expanded_tokens):")
print(expanded_tokens)

# Modify one token in the expanded tensor
expanded_tokens[0, 0, 0] = 999
print("\nExpanded tensor after modification (expanded_tokens):")
print(expanded_tokens)

# Show that the change affects all replicated views
print("\nOriginal tensor after modification (latent_tokens):")
print(latent_tokens)

# Reset original tensor
latent_tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Repeat the tensor
repeated_tokens = latent_tokens.unsqueeze(0).repeat(B, 1, 1)
print("\nRepeated tensor (repeated_tokens):")
print(repeated_tokens)

# Modify one token in the repeated tensor
repeated_tokens[0, 0, 0] = 999
print("\nRepeated tensor after modification (repeated_tokens):")
print(repeated_tokens)

# Show that the change does not affect the original tensor or other copies
print("\nOriginal tensor after modification (latent_tokens):")
print(latent_tokens)
