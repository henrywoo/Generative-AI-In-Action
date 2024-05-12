# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
import torch
import matplotlib.pyplot as plt

def positional_encoding(max_position, d_model, min_freq=1e-4):
    position = torch.arange(max_position, dtype=torch.float32)
    mask = torch.arange(d_model)
    sin_mask = (mask % 2).float()
    cos_mask = 1 - sin_mask
    exponent = 2 * (mask // 2)
    exponent = exponent.float() / d_model
    freqs = min_freq ** exponent
    angles = position[:, None] * freqs[None, :]
    pos_enc = torch.cos(angles) * cos_mask + torch.sin(angles) * sin_mask
    return pos_enc

### Plotting
d_model = 128
max_pos = 256
pos_enc_matrix = positional_encoding(max_pos, d_model).numpy()  # Convert to NumPy array for plotting

plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
plt.pcolormesh(pos_enc_matrix, cmap='viridis')
plt.xlabel('Depth')
plt.xlim((0, d_model))
plt.ylabel('Position')
plt.title("Positional Encoding Matrix Heat Map")
plt.colorbar()
plt.show()
