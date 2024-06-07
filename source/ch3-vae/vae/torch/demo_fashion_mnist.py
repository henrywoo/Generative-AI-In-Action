import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision
torch.manual_seed(0)
here = os.path.abspath(os.path.dirname(__file__))

# Define the class names
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the data to [-1, 1]
])

# Download and load the training data
trainset = datasets.FashionMNIST(root=f'{here}/data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST(root=f'{here}/data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Get a batch of training data
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Use the built-in next function

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.grid(False)
    plt.savefig("fashion_mnist_demo.png")
    plt.show()

# Show images
imshow(torchvision.utils.make_grid(images))
# Print labels
print(','.join('%5s' % classes[labels[j]] for j in range(8)))
