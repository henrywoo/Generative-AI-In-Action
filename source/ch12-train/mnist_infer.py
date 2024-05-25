import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Reuse the same Net class defined in the training script
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    model.eval()
    return model

def predict(image_path, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ensure device consistency
    transform = transforms.Compose([
        transforms.Grayscale(),  # Ensure image is grayscale
        transforms.Resize((28, 28)),  # Resize to match MNIST dimensions
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Move the image tensor to the device
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)
    return pred.item()


def main():
    model = load_model()
    image_path = '4.png'  # Change this to your image path
    prediction = predict(image_path, model)
    print(f'Predicted Digit: {prediction}')

if __name__ == '__main__':
    main()
