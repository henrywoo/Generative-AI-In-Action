import numpy as np
import matplotlib.pyplot as plt

# Define transformation matrices
def scaling_matrix(k):
    return np.array([
        [k, 0],
        [0, k]
    ])

def unequal_scaling_matrix(k1, k2):
    return np.array([
        [k1, 0],
        [0, k2]
    ])

def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def shear_matrix(k):
    return np.array([
        [1, k],
        [0, 1]
    ])

# Add a vertical shear matrix function
def vertical_shear_matrix(k):
    return np.array([
        [1, 0],
        [k, 1]
    ])

def hyperbolic_rotation_matrix(phi):
    return np.array([
        [np.cosh(phi), np.sinh(phi)],
        [np.sinh(phi), np.cosh(phi)]
    ])

# Function to plot the transformations
def plot_transformations(transformations, titles):
    plt.figure(figsize=(8, 6))
    for i, (transformation, title) in enumerate(zip(transformations, titles)):
        plt.subplot(2, 3, i + 1)
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.grid(True, which='both')

        original_shape = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
        transformed_shape = transformation @ original_shape

        plt.plot(original_shape[0], original_shape[1], 'b-', label='Original', alpha=0.5, linewidth=2)
        plt.plot(transformed_shape[0], transformed_shape[1], 'r-', label='Transformed', alpha=0.5, linewidth=2)

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
        determinant = np.linalg.det(transformation)
        plt.title(f"{title}\n(Determinant: {determinant:.2f})", fontsize=8)
        plt.legend()

    plt.tight_layout(pad=1.0, w_pad=0.1, h_pad=2.0)
    plt.savefig('matrix_transformations.png')
    plt.show()

# Define transformations
k = 1.5
k1, k2 = 1.5, 0.5
theta = np.radians(45)
shear_factor = 0.5
phi = np.radians(45)  # Angle for hyperbolic rotation
vertical_shear_factor = 0.8

transformations = [
    scaling_matrix(k),
    unequal_scaling_matrix(k1, k2),
    rotation_matrix(theta),
    shear_matrix(shear_factor),
    vertical_shear_matrix(vertical_shear_factor),
    hyperbolic_rotation_matrix(phi)
]

titles = [
    'Scaling',
    'Unequal Scaling',
    'Rotation',
    'Horizontal Shear',
    'Vertical Shear',
    'Hyperbolic Rotation'
]

# Plot transformations
plot_transformations(transformations, titles)
