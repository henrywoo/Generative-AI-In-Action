import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the same CNN architecture as used in training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_model_v2.pth', map_location=device))
    model.eval()
    return model

def predict(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure image is grayscale
        transforms.Resize((28, 28)),  # Resize to match MNIST dimensions
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using the same parameters as during training
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Prepare image tensor and move to the correct device
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)
    return pred.item()

def main():
    model = load_model()
    image_path = '4.png'  # Change this to the path of your test image
    prediction = predict(image_path, model)
    print(f'Predicted Digit: {prediction}')

if __name__ == '__main__':
    main()
