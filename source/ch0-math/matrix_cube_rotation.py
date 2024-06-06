import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to create the rotation matrix using Rodrigues' rotation formula
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    matrix = np.array([
        [cos_theta + x*x*(1 - cos_theta), x*y*(1 - cos_theta) - z*sin_theta, x*z*(1 - cos_theta) + y*sin_theta],
        [y*x*(1 - cos_theta) + z*sin_theta, cos_theta + y*y*(1 - cos_theta), y*z*(1 - cos_theta) - x*sin_theta],
        [z*x*(1 - cos_theta) - y*sin_theta, z*y*(1 - cos_theta) + x*sin_theta, cos_theta + z*z*(1 - cos_theta)]
    ])
    return matrix

# Function to apply the rotation to a set of points
def rotate(points, axis, theta):
    R = rotation_matrix(axis, theta)
    return np.dot(points, R.T)

# Define a cube with vertices
cube = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Define the edges of the cube for plotting
edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
]

# Colors for each vertex
vertex_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

# Rotation parameters
axis = np.array([1, 1, 0])  # Rotate about the vector (1, 1, 0)

# Function to update the plot
def update_plot(frame):
    theta = frame * np.pi / 180  # Convert frame to radians
    ax.cla()
    ax.set_title(f'Rotated Cube\nTheta: {theta:.2f} radians')

    # Rotate the cube
    rotated_cube = rotate(cube, axis, theta)

    # Debug: print rotated vertices
    print(f'Rotated vertices for theta={theta:.2f} radians:\n{rotated_cube}\n')

    # Plot the rotated cube with colored vertices
    for i, edge in enumerate(edges):
        ax.plot3D(*zip(*rotated_cube[edge]), color=vertex_colors[i % len(vertex_colors)], marker='o')

    # Plot vertices with different colors
    for i, vertex in enumerate(rotated_cube):
        ax.scatter(*vertex, color=vertex_colors[i], s=100)

    # Set axis limits and labels
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Create a figure and axis for plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original cube for reference
for i, edge in enumerate(edges):
    ax.plot3D(*zip(*cube[edge]), color=vertex_colors[i % len(vertex_colors)], marker='o')

# Plot vertices with different colors
for i, vertex in enumerate(cube):
    ax.scatter(*vertex, color=vertex_colors[i], s=100)

# Create an animation
ani = FuncAnimation(fig, update_plot, frames=np.arange(0, 360, 1), interval=20)

# Display the plot
plt.show()
