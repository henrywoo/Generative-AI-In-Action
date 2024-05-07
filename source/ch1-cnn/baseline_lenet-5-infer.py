import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Load the trained model
model = LeNet5()
model.load_state_dict(torch.load("model_lenet5.pth"))
model.eval()
# Define the image transformation
transform = transforms.Compose([
    transforms.Grayscale(),            # Convert to grayscale
    transforms.Resize((32, 32)),       # Resize to 32x32
    transforms.ToTensor(),             # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the image
])

# Load and preprocess the image
image_path = '3.png'
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension

# Display the image
plt.imshow(image.squeeze(0).numpy().squeeze(), cmap='gray')
plt.title("Processed Image")
plt.show()

# Predict the class of the image
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Predicted class:", predicted.item())

