from fastai.vision.all import *
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
path = untar_data(URLs.CIFAR)
dls = ImageDataLoaders.from_folder(path, valid='test')

# Function to display images
def show_images(dls, num_images):
    dls.show_batch(max_n=num_images, figsize=(10, 10))

show_images(dls, 9)
plt.show()

