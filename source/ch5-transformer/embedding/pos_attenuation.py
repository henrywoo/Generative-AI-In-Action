import numpy as np
import matplotlib.pyplot as plt

def rope_attenuation(d_model, max_position):
    position_ids = np.arange(max_position)
    freqs = 1.0 / (10000 ** (np.arange(d_model // 2) / (d_model // 2)))  # Angular frequencies
    angles = position_ids[:, np.newaxis] * freqs[np.newaxis, :]

    # Properly expand dimensions to compute angle differences
    angles_i = angles[:, np.newaxis, :]
    angles_j = angles[np.newaxis, :, :]

    # Compute the absolute differences between cosines of angle differences
    cos_diff = np.abs(np.cos(angles_i - angles_j))
    return cos_diff

# Example for d_model = 4 (2D RoPE), max_position = 10
d_model = 4
max_position = 50
attenuation_matrix = rope_attenuation(d_model, max_position)

# Plot the attenuation matrix
plt.imshow(attenuation_matrix[:, :, 0], cmap='viridis', aspect='auto')
plt.colorbar(label='Attenuation')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('RoPE Attenuation Matrix')
plt.show()
