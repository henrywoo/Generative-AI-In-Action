import torch

# Define vectors A and B
A = torch.tensor([1, 2, 3], dtype=torch.float32)
B = torch.tensor([4, 5, 6], dtype=torch.float32)

# Reshape vectors for matrix multiplication: A as 1x3, B as 3x1
A_reshaped = A.view(1, -1)  # Makes A a 1x3 vector
B_reshaped = B.view(-1, 1)  # Makes B a 3x1 vector

# Calculate cosine similarity using matrix multiplication
A_norm = A_reshaped / torch.norm(A)
B_norm = B_reshaped / torch.norm(B)
cosine_similarity = A_norm @ B_norm  # Matrix multiplication for dot product
print("Cosine Similarity:", cosine_similarity.item())


#####################################
import matplotlib.pyplot as plt

# Convert torch tensors to numpy arrays for plotting
A_np = A.numpy()
B_np = B.numpy()

# Create a new figure for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot vector A
ax.quiver(0, 0, 0, A_np[0], A_np[1], A_np[2], color='r', label='Vector A', linewidth=1.5, arrow_length_ratio=0.1)

# Plot vector B
ax.quiver(0, 0, 0, B_np[0], B_np[1], B_np[2], color='b', label='Vector B', linewidth=1.5, arrow_length_ratio=0.1)

# Setting the axes properties
ax.set_xlim([0, max(A_np[0], B_np[0]) + 1])
ax.set_ylim([0, max(A_np[1], B_np[1]) + 1])
ax.set_zlim([0, max(A_np[2], B_np[2]) + 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Plot')

# Add a legend
ax.legend()

# Show the plot
plt.show()
