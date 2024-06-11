import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F


# Function to apply k-means quantization
def quantize_image_kmeans(image, num_colors):
    # Convert image to numpy array
    image_array = np.array(image)
    w, h, d = image_array.shape
    # Reshape array to 2D
    image_array_2d = image_array.reshape(-1, d)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(image_array_2d)
    labels = kmeans.labels_
    palette = kmeans.cluster_centers_

    # Recreate the image with the palette
    quantized_image_array = palette[labels].reshape(w, h, d).astype('uint8')
    return Image.fromarray(quantized_image_array)


# Function to calculate the residual
def calculate_residual(original, quantized):
    original_array = np.array(original)
    quantized_array = np.array(quantized)
    residual = original_array - quantized_array
    return Image.fromarray(residual)


# Function to quantize the residual
def quantize_residual(residual, levels):
    residual_array = np.array(residual)
    step = 256 // levels
    quantized_residual = (residual_array // step) * step
    return Image.fromarray(quantized_residual.astype('uint8'))


# Function to calculate commitment loss
def commitment_loss(original, quantized, beta=0.25):
    original_tensor = torch.tensor(np.array(original), dtype=torch.float32)
    quantized_tensor = torch.tensor(np.array(quantized), dtype=torch.float32)
    loss = beta * F.mse_loss(original_tensor, quantized_tensor)
    return loss.item()


# Function to calculate reconstruction loss
def reconstruction_loss(original, reconstructed):
    original_tensor = torch.tensor(np.array(original), dtype=torch.float32)
    reconstructed_tensor = torch.tensor(np.array(reconstructed), dtype=torch.float32)
    loss = F.mse_loss(original_tensor, reconstructed_tensor)
    return loss.item()


# Function to reconstruct the image
def reconstruct_image(quantized, quantized_residual):
    quantized_array = np.array(quantized)
    residual_array = np.array(quantized_residual)
    reconstructed_array = quantized_array + residual_array
    reconstructed_array = np.clip(reconstructed_array, 0, 255)
    return Image.fromarray(reconstructed_array.astype('uint8'))


# Load an image
image_path = 'images/ts.jpg'
image = Image.open(image_path).convert('RGB')
image_name, image_ext = os.path.splitext(os.path.basename(image_path))

# Apply k-means quantization
num_colors = 64  # Number of colors for k-means clustering
quantized_image = quantize_image_kmeans(image, num_colors)

# Calculate residual
residual_image = calculate_residual(image, quantized_image)

# Quantize the residual
levels = 16  # Increase levels for better residual quantization
quantized_residual_image = quantize_residual(residual_image, levels)

# Compute commitment loss (between original and quantized image)
commitment_loss_value = commitment_loss(image, quantized_image)
print(f"Commitment Loss: {commitment_loss_value}")

# Reconstruct the final image
reconstructed_image = reconstruct_image(quantized_image, quantized_residual_image)

# Compute reconstruction loss (between original and reconstructed image)
reconstruction_loss_value = reconstruction_loss(image, reconstructed_image)
print(f"Reconstruction Loss: {reconstruction_loss_value}")

# Ensure the images folder exists
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

# Save the images with dynamic names
image.save(os.path.join(output_folder, f'{image_name}_original{image_ext}'))
quantized_image.save(os.path.join(output_folder, f'{image_name}_quantized{image_ext}'))
residual_image.save(os.path.join(output_folder, f'{image_name}_residual{image_ext}'))
quantized_residual_image.save(os.path.join(output_folder, f'{image_name}_quantized_residual{image_ext}'))
reconstructed_image.save(os.path.join(output_folder, f'{image_name}_reconstructed{image_ext}'))

