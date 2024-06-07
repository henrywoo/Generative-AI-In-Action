import numpy as np
import matplotlib.pyplot as plt

def scale_theta(theta_, position=1, embedding_dim=2):
    #return theta_ * (1.2 ** (-2 * (position / 1.4) / embedding_dim))
    return theta_ * position

def rope(theta_, position, embedding_dim, embedding):
    theta = scale_theta(theta_, position, embedding_dim)
    rot_mat = np.zeros((embedding_dim, embedding_dim))

    for i in range(0, embedding_dim, 2):
        rot_mat[i, i] = np.cos(theta)
        rot_mat[i, i + 1] = -np.sin(theta)
        rot_mat[i + 1, i] = np.sin(theta)
        rot_mat[i + 1, i + 1] = np.cos(theta)

    return rot_mat @ embedding, theta

embedding_dim = 2
theta = 0.2
positions = range(1, 10)
base_embedding = np.array([1, 0])

rotated_embeddings = []
rotation_angles = []
for pos in positions:
    rotated, angle = rope(theta, pos, embedding_dim, base_embedding)
    rotated_embeddings.append(rotated)
    rotation_angles.append(angle)

rotated_embeddings = np.vstack(rotated_embeddings)

colors = plt.cm.viridis(np.linspace(0, 1, len(positions)))

plt.figure(figsize=(8, 6))
plt.style.use('ggplot')

for i, pos in enumerate(positions):
    angle_degrees = np.degrees(rotation_angles[i])
    plt.quiver([0], [0], rotated_embeddings[i, 0], rotated_embeddings[i, 1],
               color=colors[i], scale=30, angles='xy', scale_units='xy',
               label=f'Pos {pos} ({angle_degrees:.2f}°)', alpha=0.8)

plt.quiver(0, 0, base_embedding[0], base_embedding[1],
           color='red', scale=30, angles='xy', scale_units='xy', label='Original', alpha=0.5)

# Setting axis limits dynamically based on data
plt.xlim(min(rotated_embeddings[:, 0])*1.1, max(rotated_embeddings[:, 0])*1.1)
plt.ylim(min(rotated_embeddings[:, 1])*1.1, max(rotated_embeddings[:, 1])*1.1)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Rotary Positional Embeddings (RoPE)")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True)
plt.axis('equal')

# Adjust layout to remove extra whitespace
plt.tight_layout()

plt.savefig("rotary.png")
plt.show()
