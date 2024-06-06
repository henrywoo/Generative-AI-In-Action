import numpy as np
import matplotlib.pyplot as plt


# Define the rotation matrix for a given angle theta (in radians)
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


# Function to plot the vectors
def plot_vectors(vectors, colors, labels, title):
    with plt.style.context('ggplot'):
        plt.figure()
        ax = plt.gca()
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        ax.grid(True, which='both')

        for i, v in enumerate(vectors):
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])

        ax.set_aspect('equal', 'box')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.legend()
        plt.title(title)
        plt.savefig('matrix_rot_45.png')
        plt.show()


if __name__ == '__main__':
    # Define the original vector
    original_vector = np.array([1, 0])

    # Define the angle of rotation (45 degrees)
    theta = np.radians(45)

    # Calculate the rotated vector
    rot_matrix = rotation_matrix(theta)
    rotated_vector = np.dot(rot_matrix, original_vector)

    # Plot the original and rotated vectors
    plot_vectors(
        [original_vector, rotated_vector],
        ['blue', 'red'],
        ['Original Vector', 'Rotated Vector'],
        'Rotation of a Vector by 45 Degrees'
    )
