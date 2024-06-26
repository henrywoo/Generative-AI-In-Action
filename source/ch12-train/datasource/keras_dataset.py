from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Function to display images
def show_images(images, labels, num_images):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(int(labels[i]))
        plt.axis("off")

show_images(train_images, train_labels, 25)
plt.show()

