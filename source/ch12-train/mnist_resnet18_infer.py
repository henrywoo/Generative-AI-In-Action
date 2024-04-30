import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
import torch.nn as nn

# Define the modified ResNet18 model
def modify_resnet18():
    model = resnet18(pretrained=False)  # Ensure pretrained=False since we are loading custom weights
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Output layer for 10 classes
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),  # Resizing to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalizing with MNIST mean and std
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    return image_tensor.to(device)

def predict_image(model, image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():  # No need to track gradients for inference
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  # Get the index of the max log-probability
        return predicted.item()


if __name__ == '__main__':
    # Load the trained model
    model = modify_resnet18()
    model.load_state_dict(torch.load('mnist_resnet18_best.pt'))
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Example usage
    image_path = '3.png'
    predicted_digit = predict_image(model, image_path)
    print(f'The predicted digit is: {predicted_digit}')
