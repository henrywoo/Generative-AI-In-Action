import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

# Sample 2D vectors
v1 = np.array([2, 7])
v2 = np.array([4, 6])  # Similar to v1
v3 = np.array([-3, 1])  # Somewhat orthogonal to v1
v4 = np.array([-4, -5]) # Opposite direction to v1

# Calculate similarities
sim_v1_v2 = cosine_similarity(v1, v2)
sim_v1_v3 = cosine_similarity(v1, v3)
sim_v1_v4 = cosine_similarity(v1, v4)

# Plot setup
plt.figure(figsize=(6, 6))
plt.style.use('ggplot')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Vector Similarity Visualization")

# Plotting the vectors
scale_factor = 18
plt.quiver(0, 0, v1[0], v1[1], color='blue', scale=scale_factor)
plt.quiver(0, 0, v2[0], v2[1], color='green', scale=scale_factor)
plt.quiver(0, 0, v3[0], v3[1], color='red', scale=scale_factor)
plt.quiver(0, 0, v4[0], v4[1], color='orange', scale=scale_factor)

# Adding text labels next to the vectors at the arrow tips
plt.text(v1[0] + 0.5, v1[1] - 0.5, 'v1', color='blue', fontsize=10, verticalalignment='bottom', horizontalalignment='left')
plt.text(v2[0] + 0.5, v2[1] - 0.5, 'v2', color='green', fontsize=10, verticalalignment='bottom', horizontalalignment='left')
plt.text(v3[0] - 0.5, v3[1] + 0.5, 'v3', color='red', fontsize=10, verticalalignment='top', horizontalalignment='right')
plt.text(v4[0] - 0.5, v4[1] + 0.5, 'v4', color='orange', fontsize=10, verticalalignment='top', horizontalalignment='right')

# Adding text annotations with similarity results
plt.text(-9, 8, f"Similarity (v1, v2): {sim_v1_v2:.2f}", fontsize=9, color='green')
plt.text(-9, 7, f"Similarity (v1, v3): {sim_v1_v3:.2f}", fontsize=9, color='red')
plt.text(-9, 6, f"Similarity (v1, v4): {sim_v1_v4:.2f}", fontsize=9, color='orange')

plt.grid(True)
plt.savefig("similarity_2d.png")
plt.show()
