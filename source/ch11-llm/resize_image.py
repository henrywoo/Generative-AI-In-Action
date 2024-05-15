import os
from PIL import Image

# Define the target width
TARGET_WIDTH = 600

# Get a list of all files in the current directory
files = os.listdir('.')

# Filter out image files (you can extend the list of extensions if needed)
image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]


# Function to resize an image
def resize_image(image_path):
    with Image.open(image_path) as img:
        # Check if the image width is greater than the target width
        if img.width > TARGET_WIDTH:
            # Calculate the new height to maintain aspect ratio
            aspect_ratio = img.height / img.width
            new_height = int(TARGET_WIDTH * aspect_ratio)

            # Resize the image
            img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS)

            # Save the image (overwrite the original file)
            img.save(image_path)
            print(f"Resized: {image_path}")


# Resize all images in the list
for image_file in image_files:
    resize_image(image_file)

print("Image resizing complete.")
