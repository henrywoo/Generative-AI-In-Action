import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

# Load your image
image = plt.imread('ford.jpeg')

# Read OCR data from "button.js"
with open('ford.js', 'r') as f:
    data = json.load(f)

# Get image dimensions
height, width, _ = image.shape  # _ discards the number of channels (e.g., 3 for RGB)

# Create a figure with the correct size
fig, ax = plt.subplots(figsize=(width / 100, height / 100))
ax.imshow(image)

# Iterate through the OCR results
print(len(data))
for item in data:
    x, y = item['vertices'][0]  # Extract coordinates of top-left corner
    w = item['vertices'][2][0] - x  # Calculate width
    h = item['vertices'][2][1] - y  # Calculate height

    # Create a rectangle patch
    rect = mpatches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    # Add text annotation
    ax.text(x, y - 10, item['text'], color='blue', fontsize=8)

plt.show()

# https://app.nanonets.com/api/v2/RawOcrResponse?s3_image_path=nanonets/uploadedfiles/3243c724-c710-4e23-93b8-82fc6632e4ea/PredictionImages/ffa541d0-63e3-4b85-8893-d61ff9b9437d.jpeg
