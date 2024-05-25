import torch
import requests
from torchvision import models, transforms
from PIL import Image

def load_imagenet_classes():
    """ Load ImageNet classes from a URL into a list """
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    response = requests.get(url)
    return response.json()

# Load the pre-trained AlexNet model
model = models.alexnet(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),               # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),           # Crop the image to 224x224 pixels
    transforms.ToTensor(),                # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])
])

# Load ImageNet class labels
labels = load_imagenet_classes()

# Load the image
image_path = '5.png'
image = Image.open(image_path).convert("RGB")

# Preprocess the image
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension

# Predict the class of the image
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)  # Get the index of the highest log-probability

# Print the predicted class index and label
predicted_class_index = predicted.item()
predicted_class_label = labels[predicted_class_index]
print(f"Predicted class index: {predicted_class_index}, label: {predicted_class_label}")
