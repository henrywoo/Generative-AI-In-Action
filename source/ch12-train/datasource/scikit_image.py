from skimage import data
import matplotlib.pyplot as plt

# Load sample images
images = [data.astronaut(), data.camera(), data.checkerboard(), data.coffee(), data.coins()]

# Function to display images
def show_images(images, num_images):
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(images[:num_images]):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.axis("off")

show_images(images, 5)
plt.show()
