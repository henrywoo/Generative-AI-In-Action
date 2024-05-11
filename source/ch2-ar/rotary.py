import numpy as np
import matplotlib.pyplot as plt

def scale_theta(theta_, position=1, embedding_dim=2):
    return theta_ * (1.2 ** (-2 * (position / 1.4) / embedding_dim))

def rope(theta_, position, embedding_dim, embedding):
    theta = scale_theta(theta_, position, embedding_dim)
    rot_mat = np.zeros((embedding_dim, embedding_dim))

    for i in range(0, embedding_dim, 2):
        rot_mat[i, i] = np.cos(theta)
        rot_mat[i, i + 1] = -np.sin(theta)
        rot_mat[i + 1, i] = np.sin(theta)
        rot_mat[i + 1, i + 1] = np.cos(theta)

    return rot_mat @ embedding

# Visualization Example:
embedding_dim = 2  # 2D embedding for visualization
theta = 2.5       # Base rotation angle
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
buffer = 0.1  # Increase buffer size if necessary for better visibility

# Define a list of colors for different positions
colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

# Plot
plt.figure(figsize=(11, 8))
plt.style.use('ggplot')

print(rotated_embeddings)

# Plot each arrow with a different color and label
for i, pos in enumerate(positions):
    plt.quiver([0], [0], rotated_embeddings[i, 0], rotated_embeddings[i, 1],
               color=colors[i], scale=30, angles='xy', scale_units='xy',
               label=f'Pos{pos}', alpha=0.8)
    plt.text(rotated_embeddings[i, 0]*1.1, rotated_embeddings[i, 1]*1.1, f"{pos}", color=colors[i])

# Plot the original embedding
plt.quiver(0, 0, base_embedding[0], base_embedding[1],
           color='red', scale=30, angles='xy', scale_units='xy', label='Original', alpha=0.5)

plt.xlim(min(all_x_values) - buffer, max(all_x_values) + buffer)
plt.ylim(min(all_y_values) - buffer, max(all_y_values) + buffer)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Rotary Positional Embeddings (RoPE) Visualization")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True)
plt.axis('equal')
plt.savefig("rope.png")
plt.show()
