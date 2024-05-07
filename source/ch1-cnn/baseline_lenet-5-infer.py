import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging

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

def load_model(model_path):
    model = LeNet5()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    from hiq.vis import print_model
    print_model(model)
    return model

def preprocess_image(image_path, resize):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        image = Image.open(image_path)
    except IOError as e:
        logging.error(f"Unable to open image. {e}")
        raise
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def display_image(image):
    plt.imshow(image.squeeze(0).numpy().squeeze(), cmap='gray')
    plt.title("Processed Image")
    plt.show()

def predict(model, image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def main(args):
    logging.basicConfig(level=logging.INFO)
    model = load_model(args.model_path)
    image = preprocess_image(args.image_path, args.resize)
    display_image(image)
    prediction = predict(model, image)
    logging.info(f"Predicted class: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LeNet5 Image Classifier')
    parser.add_argument('--image_path', type=str, default='3.png', help='Path to the image file to be processed')
    parser.add_argument('--model_path', type=str, default='model_lenet5_best.pth', help='Path to the trained model file')
    parser.add_argument('--resize', type=int, default=32, help='Size to which the image will be resized (square dimensions)')
    args = parser.parse_args()
    main(args)
