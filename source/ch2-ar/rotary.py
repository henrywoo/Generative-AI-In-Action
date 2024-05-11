import numpy as np
import matplotlib.pyplot as plt

def rope(theta, position, embedding_dim, embedding):
    """Applies Rotary Positional Embeddings to an input embedding."""
    theta = theta * (1.2 ** (-2 * (position // 1) / embedding_dim))  # Frequency based on position
    rot_mat = np.zeros((embedding_dim, embedding_dim))

    for i in range(0, embedding_dim, 2):
        rot_mat[i, i] = np.cos(theta)
        rot_mat[i, i + 1] = -np.sin(theta)
        rot_mat[i + 1, i] = np.sin(theta)
        rot_mat[i + 1, i + 1] = np.cos(theta)

    return rot_mat @ embedding

# Visualization Example:
embedding_dim = 2  # 2D embedding for visualization
theta = 2.0       # Base rotation angle
positions = range(1, 10)  # Positions 1 to 9
base_embedding = np.array([1, 0])  # Starting embedding vector

rotated_embeddings = []
for pos in positions:
    rotated = rope(theta, pos, embedding_dim, base_embedding)
    rotated_embeddings.append(rotated)

rotated_embeddings = np.vstack(rotated_embeddings)

# Find the maximum and minimum values for setting axis limits
all_x_values = np.concatenate(([0], rotated_embeddings[:, 0]))
all_y_values = np.concatenate(([0], rotated_embeddings[:, 1]))
buffer = 0.01  # Increase buffer size if necessary

print(base_embedding)
print(rotated_embeddings)

# Plot
plt.figure(figsize=(8, 6))
plt.style.use('ggplot')
plt.quiver([0]*len(positions), [0]*len(positions), rotated_embeddings[:, 0], rotated_embeddings[:, 1],
           angles='xy', scale_units='xy', scale=30, color='blue', label='Rotated', alpha=0.5)
plt.quiver(0, 0, base_embedding[0], base_embedding[1],
           angles='xy', scale_units='xy', scale=30, color='red', label='Original', alpha=0.5)

plt.xlim(min(all_x_values) - buffer, max(all_x_values) + buffer)
plt.ylim(min(all_y_values) - buffer, max(all_y_values) + buffer)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("RoPE")
plt.legend()
plt.grid(True)
plt.axis('equal')  # Ensure that one unit in x is the same as one unit in y
plt.savefig("rope.png")
plt.show()
