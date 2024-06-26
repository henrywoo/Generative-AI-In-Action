from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the imagenette dataset with the desired configuration (e.g., '160px')
dataset = load_dataset("frgfm/imagenette", 'full_size', split='train')

# Function to display images
def show_images(dataset, num_images):
    plt.figure(figsize=(10, 10))
    for i, data in enumerate(dataset.select(range(num_images))):
        img = np.array(data['image'])
        label = data['label']
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")

show_images(dataset, 4)
plt.show()
