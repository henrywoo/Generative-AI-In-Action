import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load pre-trained model and feature extractor from Hugging Face
model_name = "microsoft/resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Function to preprocess the image for the model
def preprocess_image(images):
    images = images.permute(0, 2, 3, 1)  # Change shape to (batch_size, height, width, channels)
    images = (images + 1) / 2  # Scale images to [0, 1] range
    images = feature_extractor(images=images, return_tensors="pt").pixel_values
    images = images.to(device)
    return images

# Load MNIST dataset and filter to include only digits 4 and 9
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model's expected input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = MNIST(root='./data', train=False, download=True, transform=transform)
indices = [i for i, target in enumerate(mnist.targets) if target in [4, 9]]
mnist_4_9 = Subset(mnist, indices)
data_loader = DataLoader(mnist_4_9, batch_size=64, shuffle=False)

# Evaluate the model on the filtered dataset
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in data_loader:
        images = images.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel
        images = preprocess_image(images)
        outputs = model(images).logits
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Convert labels from 4, 9 to 0, 1 for binary classification metrics
all_labels = np.array([0 if label == 4 else 1 for label in all_labels])
all_preds = np.array([0 if pred == 4 else 1 for pred in all_preds])

# Calculate and print performance metrics
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=['4', '9'])

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
