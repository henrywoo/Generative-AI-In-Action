import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Function to display images
def show_images(dataset, num_images):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(num_images)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(int(label))
        plt.axis("off")

show_images(train_dataset, 25)
plt.show()
